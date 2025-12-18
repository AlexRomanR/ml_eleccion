from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from strawberry.fastapi import GraphQLRouter
from app.api.schema import schema

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://taskflowuagrm.netlify.app",
        "http://localhost:4200",
        "http://127.0.0.1:4200",
    ],
    allow_credentials=False,   # para GraphQL normal suele ir False
    allow_methods=["*"],
    allow_headers=["*"],
)


graphql_app = GraphQLRouter(schema)
app.include_router(graphql_app, prefix="/graphql")
@app.get("/health")
def health():
    return {"ok": True}


@app.get("/")
def root():
    return {"message": "Use /graphql endpoint"}
