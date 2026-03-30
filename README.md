# FraudSentinel AI - Backend
### Real-Time Fraud Detection for African Financial Systems

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/flask-2.3-red.svg)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](http://makeapullrequest.com)

> **BeOrchid Africa Hackathon 2026 - Top 30 Finalist** 🏆

FraudSentinel AI is a production-ready fraud detection system designed specifically for African financial institutions. It combines a 7-model ensemble machine learning approach with rule-based detection, providing real-time risk scoring with sub-200ms latency.

## Features

### Fraud Detection Engine
- **7-Model Ensemble** - Random Forest, XGBoost, LightGBM, CatBoost, Gradient Boosting, AdaBoost, Bagging
- **Rule-Based Engine** - 6 custom rules for known fraud patterns
- **Self-Learning** - Feedback loop adapts weights based on analyst input
- **Layer 3 Lite** - Real-time risk adjustment with velocity checks

### Security & Compliance
- **JWT Authentication** - Secure token-based API access
- **Role-Based Access Control** - Admin, Analyst, Investigator, Compliance, Viewer
- **Sovereign Mode** - Data never leaves Africa
- **Audit Log** - Complete transaction history

### API Endpoints
- **Authentication** - Login, Refresh token
- **User Management** - Full CRUD operations
- **Fraud Detection** - Real-time scoring, transaction history
- **Admin Controls** - User management, system settings

###  African Context
- **KES Currency** - Kenyan Shilling support
- **Nairobi Timezone** - All timestamps in EAT
- **Local Rules** - Custom rules for African fraud patterns
- **Low Latency** - Optimized for African network conditions

### Explainable AI
- **LLM-Powered Explanations** - Customer-friendly fraud explanations (Groq/Llama)
- **Rule-Based Fallback** - Always have explanations even without LLM
- **Transparent Decisions** - Clear reasoning for each risk score

###  User Management
- **Full CRUD Operations** - Create, read, update, delete users
- **Role Management** - Assign and update user roles
- **Password Reset** - Email reset or temporary password generation
- **Account Management** - Enable/disable user accounts


## Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- npm 
- Git

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/SayiaFelix/finGuardAI.git
cd development

```
### Backend setup
pip install -r requirements.txt
cp .env 

```