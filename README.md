
# VC/Founder Search Application

Coded end-to-end by: Immanuel Lewis

This repository is shared only for review purposes for my portfolio. Forking, cloning, redistribution, or reuse of any part of this code is strictly prohibited. 



A powerful AI-enhanced search engine for VC and founder blog content, featuring advanced semantic search, query enhancement, and comprehensive content analysis.

##  Overview

The VC/Founder Search application is designed to help entrepreneurs, investors, and startup enthusiasts discover relevant insights from curated VC and founder blogs. It combines traditional search with modern AI techniques to provide highly relevant results for fundraising, strategy, and startup advice. I created this project because early stage tech fouonders want to hear directly from the top VCs and Entrepreneurs, not just chatGPT. I also incorporated prompt enhancement because, the quality of your prompt, determines, the quality of your answers. 

##  Key Features

###  Advanced Search Capabilities
- **Semantic Search**: Natural language queries using sentence transformers
- **Tag-based Search**: Predefined tags for quick topic discovery
- **AI Query Enhancement**: GPT-4o-mini powered query refinement
- **Strategic Query Matching**: Pre-embedded strategic question matching

### 🤖 AI Enhancement
- **Query Reframing**: Transform surface-level questions into strategic insights
- **Multiple Enhancement Styles**: Strategic, clarity, and tactical improvements
- **LLM Integration**: OpenAI GPT-4o-mini and Claude support
- **Smart Validation**: Shadow validation for risk mitigation

###  Content Processing
- **Intelligent Scraping**: Multi-platform blog content extraction
- **Content Chunking**: Smart text segmentation with overlap
- **Embedding Generation**: Vector embeddings for semantic search
- **Tag Classification**: Automatic content categorization

###  Specialized Topics
Pre-configured for startup and VC topics including:
- SAFE notes and fundraising
- Series A/B/C strategies
- Product-market fit indicators
- Cap table and dilution
- Term sheet negotiation
- SaaS metrics and benchmarks
- Founder advice and lessons learned

## 🛠 Technology Stack

- **Backend**: Flask with comprehensive API endpoints
- **Frontend**: HTML/JavaScript interface with real-time search
- **Database**: SQLite with optimized schemas
- **AI/ML**: 
  - Sentence Transformers for embeddings
  - OpenAI GPT-4o-mini for query enhancement
  - Claude 3 (Opus/Sonnet) support
  - TikToken for accurate tokenization
- **Scraping**: 
  - BeautifulSoup4 for HTML parsing
  - Feedparser for RSS feeds
  - Platform-specific scrapers (Medium, Substack, etc.)
- **Data Processing**: Pandas, NumPy, scikit-learn

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key (for query enhancement)
- Claude API key (optional, for alternative LLM)

##  Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_claude_api_key_here  # Optional
```

### 3. Database Initialization

```bash
# Initialize the database and scrape blog content
python init_db.py
```

### 4. Run the Application

```bash
# Start the Flask backend
python main.py
```

The application will be available at `http://0.0.0.0:5000`

##  Usage Guide

### Basic Search

1. **Tag Search**: Select predefined tags like "Fundraising", "SAFE", or "Product Market Fit"
2. **Semantic Search**: Enter natural language queries like "How to negotiate with VCs"

### Advanced Features

1. **AI Enhancement**: Enable query enhancement for strategic reframing
2. **Multiple Models**: Choose between GPT-4o-mini and Claude models
3. **Enhancement Styles**: 
   - **Strategic**: Transform tactical questions into strategic insights
   - **Clarity**: Make vague questions more specific and actionable
   - **Tactical**: Focus on concrete actions and processes

### Example Queries

```
Original: "What's a SAFE?"
Strategic: "How do I structure early capital to maximize future leverage and control?"

Original: "How much should I raise?"
Strategic: "What's the minimum viable capital that creates maximum optionality?"

Original: "How do I find investors?"
Tactical: "What's the step-by-step process for building a target list of 50 relevant investors?"
```

## 📁 Project Structure

```
├── main.py                 # Application entry point
├── flask_backend.py        # Flask API server
├── blog_scraper.py         # Content scraping engine
├── streamlit_app.py        # Alternative Streamlit interface
├── pre_embed_queries.py    # Query enhancement system
├── search_interface.html   # Web frontend
├── blog_content.db         # SQLite database
├── embeddings/             # Vector embeddings storage
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Search Settings
- **Semantic Search**: Enabled by default, requires sentence-transformers
- **Enhanced Matching**: Uses pre-embedded strategic queries
- **Result Limits**: Configurable (default: 10 results)

### AI Enhancement
- **Default Model**: GPT-4o-mini (most cost-effective)
- **Fallback**: System works without API keys (enhancement disabled)
- **Rate Limiting**: Built-in request throttling

### Content Sources
The system scrapes from curated VC and founder blogs including:
- Paul Graham essays
- First Round Review
- Andreessen Horowitz blog
- Sequoia Capital insights
- Y Combinator blog
- And many more...

## 🔍 API Endpoints

### Search
- `POST /api/search/semantic` - Semantic search with AI enhancement
- `POST /api/search/tags` - Tag-based search
- `GET /api/tags` - Get available tags
- `GET /api/stats` - Database statistics

### Configuration
- `GET /api/models` - Available AI models
- `GET /` - Web interface

## 🛠 Development

### Adding New Blog Sources

1. Add blog URL to `VC_Founder_Blogs_List.csv`
2. Run scraper: `python blog_scraper.py`
3. Custom parsers can be added in `custom_blog_parsers.py`

### Debugging

```bash
# Check system status
python debug_startup.py

# Analyze scraping failures
python analyze_failures.py

# Test API connectivity
python test_openai.py
```

### Performance Optimization

The application includes several optimization features:
- Database connection pooling
- Embedding caching
- Request batching
- Intelligent retry logic

##  Database Schema

### Core Tables
- **blogs**: Blog metadata and scraping status
- **posts**: Individual blog posts with content
- **chunks**: Processed content segments with tags
- **chunk_embeddings**: Vector embeddings for semantic search

### Analytics
- Blog scraping success rates
- Content processing statistics
- Search query analysis
- Performance metrics

##  Security & Privacy

- API keys stored in environment variables
- No user data collection
- Local database storage
- Rate limiting on external APIs

##  Troubleshooting

### Common Issues

**Database locked errors**:
```bash
# Reset database
rm blog_content.db
python init_db.py
```

**Embedding errors**:
```bash
# Install/update sentence-transformers
pip install --upgrade sentence-transformers
```

**API rate limits**:
- Reduce query frequency
- Check API key quotas
- Use alternative models

### Debug Mode

```bash
# Enable verbose logging
export FLASK_DEBUG=1
python main.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Format code
black . && isort .
```

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Support

