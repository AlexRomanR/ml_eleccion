from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
from app.api.schema import schema

app = FastAPI()

# GraphQL Router - Único punto de entrada
graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")

# Redirigir root a graphql para facilidad (opcional pero útil)
@app.get("/")
def root():
    return {"message": "Use /graphql endpoint"}
