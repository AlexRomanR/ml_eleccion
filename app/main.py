# app/main.py
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from app.api.schema import schema

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

@app.get("/")
def root():
    return {"message": "Use /graphql endpoint"}
