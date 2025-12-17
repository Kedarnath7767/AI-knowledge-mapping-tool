# 📘 AI-KnowMap: Cross-Domain Knowledge Mapping Tool

AI-KnowMap is an intelligent system that discovers conceptual links across different domains using:

✔ Semantic Search  
✔ Named Entity Recognition  
✔ Relation Extraction  
✔ Interactive Knowledge Graphs  
✔ Dataset Insights & Analytics  
✔ User Authentication + Admin Tools  

Built using **Streamlit**, **spaCy**, **Sentence Transformers**, and **PyVis**.

---

# 🚀 Features

### 📤 1. Upload Any Dataset Format
Upload files in:

- CSV  
- Excel (XLS/XLSX)  
- TXT  

The system automatically:

- Detects sentence column  
- Generates ID/domain/label if missing  
- Cleans and normalizes the dataset  

---

### 🧠 2. Automatic NLP Pipeline
Every uploaded dataset is processed through:

- Named Entity Recognition (NER)  
- Relation Extraction  
- Cross-domain mapping  
- Knowledge graph data extraction  

---

### 🔍 3. Semantic Search
Powered by `sentence-transformers`:

- Generate embeddings  
- Search semantically similar sentences  
- View similarity score + domain + label  
- Fast and accurate retrieval  

---

### 🌐 4. Knowledge Graph Visualization
Built using **PyVis + NetworkX**:

- Entities = nodes  
- Relationships = edges  
- Each domain has a unique color  
- Fully interactive (drag, zoom, hover)  
- Physics engine for layout  

---

### 🧭 5. Overview Dashboard
- Total sentences  
- Unique domains  
- Label categories  
- Domain distribution graph  
- Dataset health check  

---

### 👨‍🎓 6. Student View
A simplified view for learning purposes:

- Key concepts  
- Domain summaries  
- Graph-based insights  

---

### 💬 7. Feedback System
Users can submit:

- Errors  
- Suggestions  
- Improvements  
- Comments per record ID  

Stored with timestamp and status tracking.  

---

### 🛠 8. Admin Tools
Admin role includes:

- Merge sentences  
- Delete records  
- Manage users  
- Inspect system data  
- Access feedback moderation  

Role-based access enforced.

---

### ⚙️ 9. User Preferences
Users can configure:

- Theme (Light / Dark / Auto)  
- Search results per page  
- Tooltip visibility  
- Preferences stored per user  

---

# 🧰 Tech Stack

| Component | Technology |
|----------|------------|
| Frontend/UI | Streamlit |
| NLP | spaCy |
| Embeddings | Sentence Transformers |
| Graph Visualization | PyVis + NetworkX |
| Storage | CSV / JSON / Pickle |
| Deployment | Docker + Cloud VM |
| Authentication | Custom JSON-based |

---

# 📤 Dataset Format (Flexible)

Works with datasets containing:

| Column | Description |
|--------|-------------|
| id | Unique row ID |
| sentence | Text content |
| domain | Subject category |
| label | Relation type |

If your dataset does **NOT** include these columns —  
The system will **automatically detect and generate them**.

---

# 🔍 Using Semantic Search

1. Go to **Semantic Search**  
2. If embeddings don't exist → click *Generate Embeddings*  
3. Enter your query  
4. View:

- Closest 3 matches  
- Domains  
- Labels  
- Similarity scores  

---

# 🌐 Knowledge Graph Usage

Click **Generate Knowledge Graph**:

- Extracts entities  
- Builds relationship triples  
- Visualizes with domain colors  
- Interactive UI  

Graph auto-saves to:  
`knowledge_graph.html`

---

# 💬 Feedback Section

Users can:

- Enter record ID  
- Choose feedback type  
- Add comments  
- Submit feedback  

Admins can:

- Mark status  
- View all feedback  
- Export feedback  

---

# 🛠 Admin Tools (Only for Admin Role)

- Merge duplicate nodes (sentences)  
- Delete dataset records  
- Manage user roles  
- View user list  
- System-level insights  

---

# ☁️ Cloud Deployment (Docker + VM)

### Build Docker Image
```bash
docker build -t knowmap .
```

### Run Container
```bash
docker run -p 8501:8501 knowmap
```

### Access App
```
http://<your-public-ip>:8501
```

---

# 🧪 Testing & Optimization Notes

During evaluation, test:

✔ Dataset upload  
✔ NLP extraction  
✔ Graph generation  
✔ Semantic search performance  

Optimize by:

- Cleaning Docker layers  
- Reducing model load time  
- Minimizing logs  
- Improving graph generation speed  

---

# 🚀 Future Enhancements

- Transformer-based relation extraction  
- Real-time streaming knowledge graph  
- Custom domain-specific embeddings  
- Multi-language support  
- Dedicated admin dashboard  
- Exportable analytical reports  

---

# 👨‍💻 Author
Developed by **<Your Name>**  
For academic and research purposes.
