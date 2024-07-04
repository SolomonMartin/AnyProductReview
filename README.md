# AnyProductReview

AnyProductReview is a LangChain and Streamlit application that leverages Tavily Search and Google Generative AI to provide comprehensive product reviews. Users can input a product name, and the app will fetch reviews, analyze them, and provide a detailed summary, highlighting advantages, disadvantages, special features, and value for money based on the collected reviews.

## Features

- **Product Review Fetching:** Utilizes Tavily Search to fetch reviews for the given product name.
- **Text Splitting and Embedding:** Splits the fetched reviews into manageable chunks and creates embeddings using Google Generative AI.
- **Review Analysis:** Analyzes the reviews to provide a detailed summary including advantages, disadvantages, special features, and value for money.
- **User-Friendly Interface:** Built with Streamlit, providing a simple and intuitive user experience.

## Technologies Used

- **LangChain:** Integrates various language processing tasks such as text splitting, embedding, and review analysis.
- **Streamlit:** Creates interactive web applications directly from Python scripts.
- **Tavily Search:** Fetches product reviews from the web.
- **Google Generative AI:** Provides advanced text generation and embedding capabilities.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ProductReviewApp.git
   cd ProductReviewApp

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
3.Set up your API keys:

  Create a .env file in the root directory of the project.
  Add your Google API key and Tavily API key to the .env file:

    ```makefile
    GOOGLE_API_KEY=your_google_api_key_here
    TAVILY_API_KEY=your_tavily_api_key_here
4. Run the Streamlit app:

    ```bash
    streamlit run app.py
Usage
```
Open the app in your browser.
Enter the name of the product you want to review.
Click on "Submit" to see the detailed summary of reviews.
