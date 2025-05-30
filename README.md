# 🚦 Traffic Flow Forecasting Using GCN-LSTM

**Authors**: **Geetansha🐸 , Rachit 🤓, Ritik 🐷**

A deep learning pipeline for **spatio-temporal traffic prediction** using Graph Convolutional Networks (GCNs) and Long Short-Term Memory (LSTM) networks. This project predicts traffic speeds based on past patterns and sensor network topology.

---

## 📁 Dataset Used

**PeMSD7 (California Performance Measurement System, District 7)**  
- `PeMSD7_V_228.csv`: Traffic speed data from 228 sensors over time  
- `PeMSD7_W_228.csv`: Distance matrix between sensors (used to build graph adjacency)  

---

## 🧠 Models Implemented

| Model        | Description                                      |
|--------------|--------------------------------------------------|
| **SVR**      | Traditional regression baseline                  |
| **LSTM**     | Sequential model for temporal learning           |
| **GRU**      | Lightweight alternative to LSTM                  |
| **1D-CNN**   | Convolutional model for temporal patterns        |
| **GCN-LSTM** | Proposed model: GCN for spatial + LSTM for time  |
