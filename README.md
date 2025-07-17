# 🚀 Graph-Powered RAG

*Enhancing Retrieval-Augmented Generation with Neo4j Knowledge Graphs*

## 🎯 Overview

**Graph-Powered RAG** is an innovative project that demonstrates how to enhance traditional Retrieval-Augmented Generation (RAG) systems by integrating Neo4j graph databases with vector embeddings. This hybrid approach combines the semantic understanding of vector search with the relationship-aware capabilities of graph databases, creating a more contextually rich and intelligent retrieval system.

## ✨ Key Features

- **🔗 Hybrid Retrieval System**: Combines vector similarity search with graph-based relationship traversal
- **📊 Knowledge Graph Construction**: Automatically transforms documents into structured knowledge graphs using LLMs
- **🧠 Intelligent Entity Extraction**: Identifies and connects entities, relationships, and concepts from text
- **🔍 Context-Aware Search**: Leverages graph relationships to provide more relevant and comprehensive answers
- **🎨 Interactive Graph Visualization**: Built-in graph visualization using yFiles widgets
- **🐳 Containerized Setup**: Easy deployment with Docker Compose

## 🛠️ Technology Stack

- **Graph Database**: Neo4j 5.22.0 with APOC plugins
- **LLM Integration**: Ollama (Llama 3.2) for graph transformation and query processing
- **Vector Embeddings**: Nomic Embed Text for semantic similarity
- **Framework**: LangChain for orchestrating the RAG pipeline
- **Visualization**: yFiles Jupyter Graphs for interactive graph exploration
- **Containerization**: Docker & Docker Compose

## 📁 Project Structure

```
graph-powered-rag/
├── enhancing_rag_with_graph.ipynb    # Main notebook with implementation
├── docker-compose.yaml               # Neo4j service configuration
├── neo4j/
│   └── Dockerfile                    # Neo4j container setup with APOC plugins
|   └── apoc-5.22.0-core.jar          
├── requirements.txt                  # Python dependencies
├── Dummy_Data.txt                    # Sample data for testing
├── data/                             # Neo4j data persistence
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Ollama running locally with Llama 3.2 and Nomic Embed Text models
- Python 3.8+ with Jupyter Notebook

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/graph-powered-rag.git
   cd graph-powered-rag
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Neo4j database**:
   ```bash
   docker-compose up -d
   ```
   This will build the custom Neo4j image from the `neo4j/Dockerfile` and start the container.

5. **Configure environment variables**:
   Create a `.env` file with your Neo4j credentials:
   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=neo4j
   NEO4J_PASSWORD=your_password
   ```

6. **Install Ollama models**:
   ```bash
   ollama pull llama3.2:latest
   ollama pull nomic-embed-text:latest
   ```

7. **Run the notebook**:
   ```bash
   jupyter notebook enhancing_rag_with_graph.ipynb
   ```

## 🎮 How It Works

### 1. **Document Processing**
The system begins by loading text documents and splitting them into manageable chunks using LangChain's RecursiveCharacterTextSplitter. Each chunk is then processed by the Llama 3.2 model to extract entities, relationships, and construct a knowledge graph in Neo4j.

### 2. **Knowledge Graph Construction**
Using the LLMGraphTransformer, the system identifies:
- **Entities**: People, organizations, locations, concepts
- **Relationships**: Connections between entities with semantic meaning
- **Properties**: Additional metadata about entities and relationships

### 3. **Dual Retrieval System**
The hybrid approach employs two complementary retrieval methods:
- **Vector Search**: Finds semantically similar content using Nomic embeddings
- **Graph Traversal**: Explores entity relationships and connections using Cypher queries

### 4. **Enhanced Question Answering**
When processing queries, the system:
- Extracts entities from the user's question
- Queries the graph database for relevant entity neighborhoods
- Performs vector similarity search for additional context
- Combines both results to generate comprehensive answers

## 🔍 Example Use Cases

The project includes sample data about the Caruso family's culinary empire, demonstrating:

- **Family Genealogy**: Understanding relationships between family members like Giovanni, Maria, Antonio, and Nonna Lucia
- **Business Networks**: Tracking restaurant ownership and locations across different cities
- **Cultural Heritage**: Preserving and exploring Sicilian culinary traditions
- **Geographic Connections**: Linking locations like Santa Caterina, Rome, and the Amalfi Coast

## 📊 Sample Queries

Try these example queries with the provided sample data:

- "Who is Nonna Lucia? Did she teach anyone about restaurants or cooking?"
- "What restaurants does the Caruso family own?"
- "Tell me about the connection between Sicily and the family's cooking traditions"
- "Where are the different family restaurants located?"

## 🎨 Visualization Features

The notebook includes interactive graph visualization that allows you to:
- **Explore Relationships**: Navigate the knowledge graph structure visually
- **Entity Analysis**: Understand connections between people, places, and concepts
- **Data Insights**: Discover patterns and relationships in your data
- **Interactive Navigation**: Click and explore different nodes and edges

## 🐳 Docker Configuration

The project uses a custom Neo4j Docker setup with:

**Custom Neo4j Image**: Built from the `neo4j/Dockerfile` with APOC plugins pre-installed
**Security Configuration**: Proper settings for APOC functions and procedures
**Data Persistence**: Volumes mapped to the `./data` directory for data preservation
**Environment Variables**: Configurable authentication and plugin settings

## 🔧 Customization

### Adding Your Own Data
1. Replace `Dummy_Data.txt` with your own text files
2. Adjust chunk size and overlap parameters in the text splitter:
   ```python
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
   ```
3. Modify entity extraction prompts for domain-specific entities

### Model Configuration
- **Switch LLM Models**: Change the Ollama model in the ChatOllama initialization
- **Adjust Parameters**: Modify temperature and other generation parameters
- **Custom Embeddings**: Use different embedding models for various languages or domains

### Graph Schema Customization
- **Entity Types**: Modify the LLMGraphTransformer to recognize domain-specific entities
- **Relationship Types**: Customize relationship extraction for your use case
- **Property Extraction**: Add custom properties to entities and relationships

## 📝 Requirements

The project requires the following Python packages:

```
langchain>=0.1.0
langchain-community>=0.0.20
langchain-ollama>=0.1.0
langchain-experimental>=0.0.50
langchain-core>=0.1.0
langchain-openai>=0.1.0
neo4j>=5.0.0
tiktoken>=0.5.0
yfiles-jupyter-graphs>=1.6.0
python-dotenv>=1.0.0
json-repair>=0.7.0
pydantic>=2.0.0
jupyter>=1.0.0
notebook>=6.0.0
tensorflow>=2.0.0
```

## 🛠️ Development Setup

For development work:

1. **GPU Support**: The notebook includes TensorFlow GPU configuration for enhanced performance
2. **Memory Management**: Configured to limit GPU memory usage to 6GB
3. **Development Tools**: Consider adding testing frameworks and code formatting tools

## 🔍 Troubleshooting

### Common Issues

**Neo4j Connection**: Ensure Docker container is running and ports are correctly mapped
**Ollama Models**: Verify that required models are downloaded and accessible
**Environment Variables**: Check that `.env` file is properly configured
**Memory Issues**: Adjust GPU memory limits or disable GPU if encountering memory errors

### Performance Optimization

**Chunk Size**: Experiment with different chunk sizes for your specific use case
**Embedding Batch Size**: Adjust batch sizes for vector operations
**Graph Queries**: Optimize Cypher queries for better performance
**Caching**: Consider implementing caching for frequently accessed data

## 🤝 Contributing

We welcome contributions! Please feel free to:
- **Report Issues**: Submit bug reports and feature requests
- **Code Contributions**: Fork the repository and submit pull requests
- **Documentation**: Help improve documentation and examples
- **Testing**: Add test cases and improve code coverage

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Neo4j** for the powerful graph database platform and APOC plugins
- **LangChain** for the comprehensive RAG framework and experimental features
- **Ollama** for providing local LLM capabilities
- **yFiles** for beautiful and interactive graph visualizations
- **The Open Source Community** for the amazing tools and libraries that make this project possible

## 📚 Further Reading

- [Neo4j Graph Database Documentation](https://neo4j.com/docs/)
- [LangChain RAG Implementation Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Ollama Model Documentation](https://ollama.ai/library)
- [Graph-based RAG Research Papers](https://arxiv.org/search/?query=graph+retrieval+augmented+generation)

*Ready to revolutionize your RAG system with graph-powered intelligence? Let's build something amazing together!* 🚀

**Star this repository if you found it helpful!** ⭐