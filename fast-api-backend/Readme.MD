# This backend will consists of several services as we gonna make it as microservices architecure.
1. auth-service --> handles login/signup related things, like storing and validating JWT access tokens
2. vector-service --> it responsibles to seed the initial vector, add new vector, or update a vector in the pgvector database
3. chat-service --> where users can chat with the LLM
4. user-service --> changing name, password, profile, theme, etc.