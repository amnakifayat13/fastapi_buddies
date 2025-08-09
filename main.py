# main.py
from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os
import cloudinary
import cloudinary.uploader
from typing import List
from dataclasses import dataclass
import tempfile
import pyttsx3
import whisper
import traceback
import subprocess
import re
import time
from rapidfuzz import process, fuzz
from agents import (
    Agent, AsyncOpenAI, Runner, OpenAIChatCompletionsModel,
    RunContextWrapper, function_tool
)
from agents.run import RunConfig

# ========== LOAD ENV & CONFIG ==========
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI not set in environment variables!")

# ========== FASTAPI APP ==========
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # change for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DATABASE SETUP ==========
client = MongoClient(MONGO_URI)
db = client["menudb"]
products = db["products"]
deals = db["deals"]
orders = db["orders"]

# ========== SERIALIZER ==========
def serialize_item(item):
    return {
        "_id": str(item["_id"]),
        "name": item["name"],
        "price": item["price"],
        "category": item["category"],
        "description": item.get("description", ""),
        "image": item.get("image", "")
    }

# ---------- product/deal endpoints (unchanged) ----------
@app.post("/products")
async def add_product(
    name: str = Form(...),
    price: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(...)
):
    image_url = upload_image_to_cloudinary(image)
    product = {
        "name": name,
        "price": price,
        "category": category,
        "description": description,
        "image": image_url,
    }
    products.insert_one(product)
    return {"msg": "Product added successfully"}

@app.get("/products")
def get_all_products():
    all_products = products.find()
    menu = {}
    for item in all_products:
        cat = item["category"]
        serialized = serialize_item(item)
        if cat not in menu:
            menu[cat] = []
        menu[cat].append(serialized)
    return menu

@app.get("/products/{product_id}")
def get_product(product_id: str):
    product = products.find_one({"_id": ObjectId(product_id)})
    if not product:
        return {"error": "Product not found"}
    return serialize_item(product)

@app.put("/products/{product_id}")
async def update_product(
    product_id: str,
    name: str = Form(...),
    price: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(None)
):
    update_data = {
        "name": name,
        "price": price,
        "category": category,
        "description": description,
    }
    if image:
        update_data["image"] = upload_image_to_cloudinary(image)
    result = products.update_one({"_id": ObjectId(product_id)}, {"$set": update_data})
    return {"updated": result.modified_count}

@app.delete("/products/{product_id}")
def delete_product(product_id: str):
    result = products.delete_one({"_id": ObjectId(product_id)})
    return {"deleted": result.deleted_count}


@app.post("/deals")
async def add_deal(
    name: str = Form(...),
    price: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(...)
):
    image_url = upload_image_to_cloudinary(image)
    deal = {
        "name": name,
        "price": price,
        "category": category,
        "description": description,
        "image": image_url,
    }
    deals.insert_one(deal)
    return {"msg": "Deal added successfully"}

@app.get("/deals")
def get_all_deals():
    all_deals = deals.find()
    deal_list = {}
    for item in all_deals:
        cat = item["category"]
        serialized = serialize_item(item)
        if cat not in deal_list:
            deal_list[cat] = []
        deal_list[cat].append(serialized)
    return deal_list

@app.get("/deals/{deal_id}")
def get_single_deal(deal_id: str):
    deal = deals.find_one({"_id": ObjectId(deal_id)})
    if not deal:
        return {"error": "Deal not found"}
    return serialize_item(deal)

@app.put("/deals/{deal_id}")
async def update_deal(
    deal_id: str,
    name: str = Form(...),
    price: str = Form(...),
    category: str = Form(...),
    description: str = Form(...),
    image: UploadFile = File(None)
):
    update_data = {
        "name": name,
        "price": price,
        "category": category,
        "description": description,
    }
    if image:
        update_data["image"] = upload_image_to_cloudinary(image)
    result = deals.update_one({"_id": ObjectId(deal_id)}, {"$set": update_data})
    return {"updated": result.modified_count}

@app.delete("/deals/{deal_id}")
def delete_deal(deal_id: str):
    result = deals.delete_one({"_id": ObjectId(deal_id)})
    return {"deleted": result.deleted_count}

# ========== ORDER MODEL endpoints (unchanged) ==========
class OrderItemModel(BaseModel):
    id: str = Field(..., alias="_id")
    name: str
    price: int
    category: str
    description: str
    image: str
    quantity: int

    class Config:
        validate_by_name = True
        arbitrary_types_allowed = True

class OrderModel(BaseModel):
    tableNumber: str
    items: List[OrderItemModel]
    totalAmount: float

@app.post("/orders")
async def create_order(order: OrderModel):
    try:
        orders.insert_one(order.model_dump(by_alias=True))
        return {"message": "Order submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- helper to upload images ----------
def upload_image_to_cloudinary(file):
    result = cloudinary.uploader.upload(file.file)
    return result["secure_url"]

# ======================= SETUP AGENTS (unchanged behavior) ======================
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

# Agent tools (same as before)
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

@function_tool
async def greet_user(wrapper: RunContextWrapper[None]) -> str:
    return "Hello! 👋 Welcome to our restaurant assistant. How may I help you today?"

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

# ===================== VOICE AGENT =====================

import os
import re
import time
import tempfile
import subprocess
import traceback
from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import pyttsx3

app = FastAPI()


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")  # use "tiny" or "small" for faster tests
whisper_model = whisper.load_model(WHISPER_MODEL_NAME)

# Where generated audio files will be stored
OUTPUT_DIR = os.path.join(os.getcwd(), "generated_audio")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def speech_to_text(audio_path: str) -> str:
    """Convert audio file to text using Whisper (auto language)."""
    result = whisper_model.transcribe(audio_path)
    return result.get("text", "").strip()

def _number_near_text(text: str, match_start: int, window_chars: int = 25):
    """Try to find a number near a matched index (lookback window)."""
    start = max(0, match_start - window_chars)
    snippet = text[start: match_start + window_chars]
    nums = re.findall(r'(\d+)', snippet)
    if nums:
        # prefer the last found number before or near the match
        return int(nums[-1])
    return None

def text_to_speech(text: str) -> str:
    """Convert text to speech and return MP3 file path in OUTPUT_DIR."""
    # Temporary WAV
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    engine = pyttsx3.init()

    # Set voice rate (speed) — lower means slower
    engine.setProperty("rate", 160)  

    voices = engine.getProperty("voices")
    if voices:
        engine.setProperty("voice", voices[0].id)
    engine.save_to_file(text, tmp_wav.name)
    engine.runAndWait()

    # Convert via ffmpeg to mp3
    timestamp = int(time.time() * 1000)
    mp3_name = f"tts_{timestamp}.mp3"
    mp3_path = os.path.join(OUTPUT_DIR, mp3_name)
    try:
        subprocess.run(["ffmpeg", "-y", "-i", tmp_wav.name, mp3_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        os.remove(tmp_wav.name)
        raise RuntimeError("ffmpeg conversion failed: " + str(e))
    finally:
        if os.path.exists(tmp_wav.name):
            os.remove(tmp_wav.name)
    return mp3_path


def find_items_in_text(user_text_lower: str):
    matches = []
    db_items = []
    for p in products.find({}, {"name": 1, "price": 1, "category": 1, "description": 1, "image": 1}):
        db_items.append(("product", p))
    for d in deals.find({}, {"name": 1, "price": 1, "category": 1, "description": 1, "image": 1}):
        db_items.append(("deal", d))

    names = [doc["name"].lower() for _, doc in db_items]
    matched_names = process.extract(user_text_lower, names, scorer=fuzz.partial_ratio, limit=5)

    for match_name, score, idx in matched_names:
        if score > 70:  # threshold adjust kar sakte hain
            kind, doc = db_items[idx]
            matches.append((doc, kind))

    return matches

def extract_table_number(user_text: str):
    """Try several regex patterns to find a table number."""
    patterns = [
        r"table\s+number\s+(\d+)",
        r"table\s+(\d+)",
        r"for\s+table\s+(\d+)",
        r"table:(\d+)"
    ]
    for pat in patterns:
        m = re.search(pat, user_text, re.IGNORECASE)
        if m:
            return m.group(1)
    # fallback: any standalone '(\d+)' if context contains word 'table' earlier
    if "table" in user_text.lower():
        nums = re.findall(r"(\d+)", user_text)
        if nums:
            return nums[-1]
    return None

@app.post("/voice-agent")
async def voice_agent(audio: UploadFile = File(...)):
    try:
        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
            tmp_audio.write(await audio.read())
            tmp_audio_path = tmp_audio.name

        # Step 1: Speech -> Text
        user_text = speech_to_text(tmp_audio_path)
        user_text_lower = user_text.lower()

        # Define greeting and order keywords
        greetings = ["hello", "hi", "hey", "salam"]
        order_keywords = ["order", "table", "quantity", "buy", "place order", "karahi", "mutton", "seekh"]

        # Step 2: Greeting check only if no order keywords found
        if any(greet in user_text_lower for greet in greetings) and not any(ok in user_text_lower for ok in order_keywords):
            agent_reply = "Hello I am fine. Have a good meal and pleasant day with your buddies! What would you like to order please?"
            tts_path = text_to_speech(agent_reply)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

        # Step 3: Extract table number
        table_number = extract_table_number(user_text)

        # Step 4: Handle total bill request
        if any(kw in user_text_lower for kw in ["bill", "total", "amount"]) and table_number:
            existing_order = orders.find_one({"tableNumber": table_number})
            if existing_order:
                total_amount = existing_order.get("totalAmount", 0)
                agent_reply = f"Total bill for table {table_number} is PKR {total_amount}."
            else:
                agent_reply = f"No order found for table {table_number}."
            tts_path = text_to_speech(agent_reply)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

        # Step 5: Handle order details request
        order_details_keywords = ["order details", "order list", "items list", "show order", "order status", "list"]
        if any(kw in user_text_lower for kw in order_details_keywords) and table_number:
            existing_order = orders.find_one({"tableNumber": table_number})
            if existing_order and existing_order.get("items"):
                items_list = existing_order["items"]
                lines = [f"Order details for table {table_number}:"]
                for item in items_list:
                    lines.append(f"{item.get('quantity', 1)} × {item.get('name', 'Unknown')} (PKR {item.get('price', 0)})")
                total_amount = existing_order.get("totalAmount", 0)
                lines.append(f"Total bill: PKR {total_amount}.")
                agent_reply = " ".join(lines)
            else:
                agent_reply = f"No order found for table {table_number}."
            tts_path = text_to_speech(agent_reply)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

        # Step 6: Place order logic (order keywords present and table number exists)
        place_order_keywords = ["order", "add", "buy", "want to order", "order karna", "karahi", "mutton", "seekh"]
        if any(kw in user_text_lower for kw in place_order_keywords) and table_number:
            found = find_items_in_text(user_text_lower)
            if not found:
                agent_reply = "Sorry, I couldn't detect the item. Please specify what you want to order."
                tts_path = text_to_speech(agent_reply)
                if os.path.exists(tmp_audio_path):
                    os.remove(tmp_audio_path)
                return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

            ordered_items = []
            total_added = 0.0
            for doc, kind in found:
                name_lower = doc["name"].lower()
                idx = user_text_lower.find(name_lower)
                qty = _number_near_text(user_text_lower, idx) or 1
                try:
                    price = float(doc.get("price", 0))
                except Exception:
                    price = 0.0
                item_total = price * qty
                total_added += item_total
                ordered_items.append({
                    "_id": str(doc.get("_id")),
                    "name": doc.get("name"),
                    "price": price,
                    "category": doc.get("category", ""),
                    "description": doc.get("description", ""),
                    "image": doc.get("image", ""),
                    "quantity": qty
                })

            existing = orders.find_one({"tableNumber": table_number})
            if existing:
                orders.update_one(
                    {"_id": existing["_id"]},
                    {"$push": {"items": {"$each": ordered_items}}, "$set": {"totalAmount": float(existing.get("totalAmount", 0)) + total_added}}
                )
            else:
                new_order = {
                    "tableNumber": table_number,
                    "items": ordered_items,
                    "totalAmount": total_added
                }
                orders.insert_one(new_order)

            items_summary = ", ".join(f"{it['quantity']}× {it['name']}" for it in ordered_items)
            agent_reply = f"Your order {items_summary} for table number {table_number} has been placed and confirmed. Please wait for a while."
            tts_path = text_to_speech(agent_reply)
            if os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

        # Step 7: If none matched, fallback response
        agent_reply = "Sorry, I couldn't understand your request clearly. Please say your order or ask for the bill."
        tts_path = text_to_speech(agent_reply)
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)
        return {"user_text": user_text, "agent_text": agent_reply, "audio_url": f"/voice-audio/{os.path.basename(tts_path)}"}

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/voice-audio/{filename}")
async def get_voice_audio(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(file_path, media_type="audio/mpeg")

# ========== START SERVER ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
