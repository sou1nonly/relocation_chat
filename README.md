🌆 City Relocation Chatbot
===========================

An AI-powered chatbot to help users find their ideal city for relocation based on personal preferences like job opportunities, lifestyle, housing, cost of living, climate, and more. It intelligently remembers previous interactions, pulls real-time web search results, and uses vector search to provide meaningful and personalized responses.

Features
-----------
- Conversational Memory
- Real-Time Search
- Context-Aware Answers
- Vector Search with Pinecone
- Built with Streamlit


Project Structure
---------------------
city-relocation-chatbot/
│
├── front_streamlit.py      # Main Streamlit app 
├── backend.py              # logical CLI implementation
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation


Create `.env` file (if want to run locally)
Add the following in a `.env` file at the root of the project:

GOOGLE_API_KEY=your_google_generative_ai_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_pinecone_index_name
SERPER_API_KEY=your_serper_api_key

Built With
--------------
- Streamlit
- LangChain
- Pinecone
- Google Generative AI
- Serper.dev
