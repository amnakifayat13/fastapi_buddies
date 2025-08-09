# ======================== AGENT CONTEXT + SETUP ==========================
from agents import (
    Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel,
    RunContextWrapper, function_tool
)
from agents.run import RunConfig
import os
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from dataclasses import dataclass

# ==================== LOAD .env AND SETUP MONGODB ========================
load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment variables!")

client = MongoClient(MONGO_URI)
db = client["menudb"]
orders = db["orders"]

# ======================= SETUP GEMINI MODEL ==============================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client,
)

run_config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)


# ===================== CONTEXT CLASS & LOAD FUNCTION =====================
@dataclass
class OrderItem:
    name: str
    quantity: int
    price: float

@dataclass
class OrderInfo:
    tableNumber: str
    items: List[OrderItem]
    totalAmount: float

@dataclass
class AllOrders:
    orders: List[OrderInfo]

def load_orders_context() -> AllOrders:
    order_docs = orders.find()
    all_orders = []
    for doc in order_docs:
        order_items = []
        for item in doc.get("items", []):
            order_items.append(OrderItem(
                name=item.get("name", "Unknown"),
                quantity=item.get("quantity", 1),
                price=item.get("price", 0.0)
            ))
        all_orders.append(OrderInfo(
            tableNumber=doc.get("tableNumber"),
            items=order_items,
            totalAmount=doc.get("totalAmount", 0)
        ))
    return AllOrders(orders=all_orders)


# ===================== ORDER AGENT TOOLS =====================
@function_tool
async def get_table_total(wrapper: RunContextWrapper[AllOrders], table_number: str) -> str:
    for order in wrapper.context.orders:
        if order.tableNumber == table_number:
            return f"Total bill for table {table_number} is PKR {order.totalAmount}"
    return f"No order found for table {table_number}."

@function_tool
async def list_all_orders(wrapper: RunContextWrapper[AllOrders]) -> str:
    if not wrapper.context.orders:
        return "No orders in the database."

    result = []
    for order in wrapper.context.orders:
        order_lines = [f"\n🪑 Table {order.tableNumber}"]
        if order.items:
            for item in order.items:
                order_lines.append(f"  🍽️ {item.name} × {item.quantity} — PKR {item.price}")
        else:
            order_lines.append("  ❌ No items found.")
        order_lines.append(f"💰 Total: PKR {order.totalAmount}")
        result.append("\n".join(order_lines))

    return "\n\n".join(result)

@function_tool
async def get_table_order_details(wrapper: RunContextWrapper[AllOrders], table_number: str) -> str:
    for order in wrapper.context.orders:
        if order.tableNumber == table_number:
            if not order.items:
                return f"🪑 Table {table_number} has no items in the order."
            lines = [f"🪑 Order details for Table {table_number}:"]
            for item in order.items:
                lines.append(f"  🍽️ {item.name} × {item.quantity} — PKR {item.price}")
            lines.append(f"\n💰 Total Bill: PKR {order.totalAmount}")
            return "\n".join(lines)
    return f"❌ No order found for table {table_number}."

@function_tool
async def place_order_tool(wrapper: RunContextWrapper[AllOrders], table_number: str, item_name: str, quantity: int = 1) -> str:
    item = db["products"].find_one({"name": {"$regex": f"^{item_name}$", "$options": "i"}})
    if not item:
        item = db["deals"].find_one({"name": {"$regex": f"^{item_name}$", "$options": "i"}})
    if not item:
        return f"❌ Item '{item_name}' not found in menu."

    try:
        price = float(item.get("price", 0))
    except ValueError:
        return "❌ Invalid price format in DB."

    total = price * quantity

    new_item = {
        "_id": str(item["_id"]),
        "name": item["name"],
        "price": price,
        "category": item.get("category", ""),
        "description": item.get("description", ""),
        "image": item.get("image", ""),
        "quantity": quantity
    }

    existing_order = orders.find_one({"tableNumber": table_number})

    if existing_order:
        existing_total = float(existing_order.get("totalAmount", 0))
        new_total = existing_total + total
        orders.update_one(
            {"_id": existing_order["_id"]},
            {"$push": {"items": new_item}, "$set": {"totalAmount": new_total}}
        )
        return f"✅ Added {quantity} × {item_name} to existing order at table {table_number}."
    else:
        order_entry = {
            "tableNumber": table_number,
            "items": [new_item],
            "totalAmount": total
        }
        orders.insert_one(order_entry)
        return f"✅ New order placed for table {table_number} with {quantity} × {item_name}."


# ===================== GREETING AGENT TOOL =====================
@function_tool
async def greet_user(wrapper: RunContextWrapper[None]) -> str:
    return "Hello! 👋 Welcome to our restaurant assistant. How may I help you today?"


# ===================== AGENT INSTANCES =====================
order_agent = Agent[AllOrders](
    name="Order Assistant",
    instructions=(
        "You are an assistant that manages restaurant orders. Use tools like 'place_order_tool' to place new orders, "
        "'get_table_total' to get table totals, 'list_all_orders' to show all, and 'get_table_order_details' for details."
    ),
    model=model,
    tools=[get_table_total, list_all_orders, get_table_order_details, place_order_tool],
)

greeting_agent = Agent[None](
    name="Greeting Assistant",
    instructions="You are a friendly assistant. If the user greets, respond using 'greet_user'.",
    model=model,
    tools=[greet_user],
)


# ===================== FASTAPI ROUTES =====================
app = FastAPI()

class UserQuery(BaseModel):
    question: str

@app.post("/agent")
async def ask_agent(user_input: UserQuery):
    try:
        question = user_input.question.lower()
        if any(greet in question for greet in ["hi", "hello", "salam", "hey"]):
            result = await Runner.run(greeting_agent, user_input.question, run_config=run_config, context=None)
        else:
            context = load_orders_context()
            result = await Runner.run(order_agent, user_input.question, run_config=run_config, context=context)

        return {"response": result.final_output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
