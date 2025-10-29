# Q-Fleet: Quantum Reservoir VRP Backend

This project is the backend API for Q-Fleet, a revolutionary quantum-classical hybrid system for solving the Vehicle Routing Problem (VRP).

Its core innovation is the use of **Quantum Reservoir Computing (QRC)** to provide **real-time, sub-second adaptation** to dynamic events like traffic jams and priority deliveries, without needing to re-run a slow optimization.

This approach is based on novel research into quantum reservoirs, such as the work by QuEra Computing (arXiv:2407.02553v1).

## Key Features

* **Quantum Reservoir Computing (QRC):** A fixed, random quantum system acts as a "reservoir" to capture complex patterns. A classical readout layer is then trained to map reservoir states to optimal routes.
* **Real-Time Adaptation:** The QRC solver can adapt to new traffic jams or add priority deliveries in under a second.
* **Hybrid Solver:** The API can intelligently switch between the ultra-fast QRC solver and a traditional (but slower) QAOA solver for full-scale optimization.
* **Self-Training:** On startup, the API automatically generates synthetic VRP data and trains the QRC model, making it instantly ready for requests.

## Deployment

This project is configured for **one-click deployment on Render.com**.

The `render.yaml` file automatically instructs Render to:
1.  Use the `free` tier.
2.  Build the application using the provided `Dockerfile`.
3.  Set all necessary environment variables.
4.  Monitor the application's health at the `/api/health` endpoint.

## Key API Endpoints

* `POST /api/optimize`: Solves a new VRP problem from scratch. Set `"use_qrc": true` in the request body to use the QRC solver.
* `POST /api/traffic-jam`: Takes a list of jammed locations and instantly returns a new, adapted route using QRC.
* `POST /api/priority-delivery`: Takes a new priority location and instantly returns a new route that includes it.
* `GET /api/health`: A simple health check endpoint used by Render to verify the service is running.
* `GET /`: Root endpoint with API status and documentation links.
