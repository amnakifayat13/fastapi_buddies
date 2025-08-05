from fastapi import FastAPI, UploadFile, File, Form, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient
from bson import ObjectId
from dotenv import load_dotenv
import os
import cloudinary
import cloudinary.uploader
from typing import List


# ========== LOAD ENV & CLOUDINARY ==========
load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

def upload_image_to_cloudinary(file):
    result = cloudinary.uploader.upload(file.file)
    return result["secure_url"]

MONGO_URI = os.getenv("MONGO_URI")

# ========== FASTAPI APP ==========
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== DATABASE SETUP ==========
client = MongoClient(MONGO_URI)
db = client["menudb"]
products = db["products"]
deals = db["deals"]
orders = db["orders"]  # NEW COLLECTION


# ========== SERIALIZER ==========
def serialize_item(item):
    return {
        "_id": str(item["_id"]),
        "name": item["name"],
        "price": item["price"],
        "category": item["category"],
        "description": item.get("description", ""),
        "image": item["image"],
    }


# ========== PRODUCTS ROUTES ==========
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


# ========== DEALS ROUTES ==========
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


# ========== ORDER MODEL ==========


class OrderItem(BaseModel):
    id: str = Field(..., alias="_id")  # Accept _id from JSON but rename to id internally
    name: str
    price: str
    category: str
    description: str
    image: str
    quantity: int

    class Config:
        allow_population_by_field_name = True  # allow using "id" internally
        arbitrary_types_allowed = True

class Order(BaseModel):
    tableNumber: str
    items: List[OrderItem]
    totalAmount: float


# ========== ORDER ROUTE ==========
@app.post("/orders")
async def create_order(order: Order):
    try:
        orders.insert_one(order.model_dump(by_alias=True))  
        return {"message": "Order submitted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

        

# ========== START SERVER ==========
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
