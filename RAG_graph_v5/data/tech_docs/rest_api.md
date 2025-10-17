# REST API: Representational State Transfer Application Programming Interface

A **REST API** (Representational State Transfer Application Programming Interface) is an architectural style for designing networked applications. REST APIs have become the dominant approach for building web services due to their simplicity, scalability, and stateless nature.

## Core Principles

### 1. Client-Server Architecture
REST enforces a separation of concerns between the user interface (client) and data storage (server), improving portability and scalability.

### 2. Statelessness
Each request from client to server must contain all information needed to understand and process the request. The server does not store client context between requests.

### 3. Cacheability
Responses must define themselves as cacheable or non-cacheable to improve efficiency and reduce server load.

### 4. Uniform Interface
REST APIs use a standardized way of communicating:
- **Resource Identification**: Each resource is identified by a URI
- **Resource Manipulation**: Clients interact with resources through representations (usually JSON or XML)
- **Self-Descriptive Messages**: Each message includes enough information to describe how to process it
- **HATEOAS**: Hypermedia as the Engine of Application State

### 5. Layered System
The architecture can be composed of hierarchical layers, with each layer only interacting with adjacent layers.

## HTTP Methods

REST APIs leverage standard HTTP methods:

### GET
Retrieve data from the server. Should be idempotent and safe (no side effects).
```
GET /api/users/123
```

### POST
Create a new resource on the server.
```
POST /api/users
Body: {"name": "John Doe", "email": "john@example.com"}
```

### PUT
Update an existing resource completely.
```
PUT /api/users/123
Body: {"name": "John Smith", "email": "john.smith@example.com"}
```

### PATCH
Partially update an existing resource.
```
PATCH /api/users/123
Body: {"email": "newemail@example.com"}
```

### DELETE
Remove a resource from the server.
```
DELETE /api/users/123
```

## Status Codes

REST APIs use HTTP status codes to indicate request outcomes:
- **2xx (Success)**: 200 OK, 201 Created, 204 No Content
- **3xx (Redirection)**: 301 Moved Permanently, 304 Not Modified
- **4xx (Client Error)**: 400 Bad Request, 401 Unauthorized, 404 Not Found
- **5xx (Server Error)**: 500 Internal Server Error, 503 Service Unavailable

## Authentication Methods

Common authentication approaches:
- **API Keys**: Simple token passed in headers or query parameters
- **OAuth 2.0**: Industry-standard authorization framework
- **JWT (JSON Web Tokens)**: Compact, self-contained token format
- **Basic Authentication**: Username and password encoded in Base64

## Best Practices

1. **Use Nouns for Resources**: `/users`, not `/getUsers`
2. **Version Your API**: `/api/v1/users`
3. **Use Plural Nouns**: `/users/123`, not `/user/123`
4. **Implement Proper Error Handling**: Return meaningful error messages
5. **Use HTTPS**: Always encrypt data in transit
6. **Rate Limiting**: Protect your API from abuse
7. **Pagination**: For large datasets, implement page-based or cursor-based pagination
8. **Documentation**: Provide clear, comprehensive API documentation (Swagger/OpenAPI)

## Advantages

- **Scalability**: Stateless nature allows easy horizontal scaling
- **Flexibility**: Client and server can be developed independently
- **Platform-Independent**: Works across different programming languages and platforms
- **Human-Readable**: Uses standard HTTP and readable URIs
- **Cacheable**: Built-in HTTP caching mechanisms improve performance

## Common Use Cases

- Mobile application backends
- Microservices communication
- Third-party integrations
- Single Page Applications (SPAs)
- IoT device communication
