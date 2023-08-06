import streamlit as st
import googleapiclient.discovery
import googleapiclient.errors
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import openai
from streamlit import session_state as ss
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(page_title="YouTube Comments Analyzer", layout="wide")

# Define the function to extract the video ID from the URL
def extract_video_id(url):
    # Define the regular expression pattern to find the video ID
    pattern = r"v=([-\w]+)"
    # Use the findall function from the re module to extract the video ID
    match = re.findall(pattern, url)
    # If a match is found, return the video ID, else return None
    if match:
        return match[0]
    else:
        return None

def text_processing(text):
    text = emoji.demojize(text)
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    special_characters = string.punctuation.replace(':', '')  # Keep ':' for emojis
    text = re.sub('[%s]' % re.escape(special_characters), '', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    processed_text = ' '.join(words)
    return processed_text

def sentiment_score(review):
    blob = TextBlob(review)
    return blob.sentiment.polarity

def create_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    for _, spine in plt.gca().spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')
    plt.show()

def generate_reply(comment, response_type, notes):
    context = f"""source: YouTube Comment
    comment: {comment}
    response type: {response_type}
    {notes}
    Chatbot Response:
    Please provide a thoughtful and {response_type.lower()} response to the YouTube comment above. Your reply should be friendly, engaging, {response_type} and appreciative of the user's feedback. Aim for a response of about 1-4 sentences.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": context}],
        temperature=0.7,  # Adjust temperature for randomness of response (higher value -> more randomness)
        max_tokens=300,  # Adjust max_tokens to control the response length
    )

    # Extract and return the generated chatbot reply
    reply = response['choices'][0]['message']['content']
    return reply


# Streamlit page
st.markdown(
    "<h1 style='text-align: center; margin-bottom: 0px; color: red; font-size: 70px;'>YouTube Comments Analyzer</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<h5 style='text-align: center; margin-top: 0px;'>Empowering Content Creators with Data Visualization and Actionable Feedback</h5>",
    unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# Create a layout with centered elements
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<h5 style='text-align: center;'>Enter YouTube Video URL</h5>", unsafe_allow_html=True)
    # Provide a sample URL for users to view
    sample_url = "https://www.youtube.com/"

    if "input_url" not in ss:
        ss["input_url"] = ""

    ss["input_url"] = st.text_input("", value=ss["input_url"], placeholder=sample_url)

    input_url = ss["input_url"]
    st.markdown("<style>.stButton>button {margin: 0 auto; display: block; width: 200px;}</style>", unsafe_allow_html=True)

    num_comments = st.selectbox("Select Number of Comments", [100, 200, 300], key="2")

    # Add a button to trigger the analysis based on the provided URL
    bt = st.button("Analyze Comments", key="1")

centered_style = """
    <style>
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    </style>
"""

st.markdown(centered_style, unsafe_allow_html=True)
if bt:
    with st.spinner('In progress...'):
        # Perform sentiment analysis and fetch data using the provided URL
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyC9x6Ut0Du-sTia8PE2lOnaqDVG7P9iMOI"  # Replace with your YouTube API key

        youtube = googleapiclient.discovery.build(api_service_name, api_version, developerKey=DEVELOPER_KEY)

        # Extract the video ID from the provided URL
        video_id = extract_video_id(input_url)

        if video_id:
            next_page_token = None
            max_results = num_comments
            total_results = 0
            comments = []
            author_channel_ids = []
            channel_verified = {}
            subscriber_channel_ids = set()

            while True:
                request = youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=min(max_results, 100),  # The maximum limit per request is 100
                    pageToken=next_page_token
                )
                response = request.execute()
                total_results += len(response['items'])

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    author_channel_id = comment['authorChannelId']['value']
                    author_name = comment['authorDisplayName']
                    like_count_per_comment = comment['likeCount']
                    author_channel_ids.append(author_channel_id)

                    # Check if the author's channel ID is present in the set of subscriber channel IDs
                    is_subscriber = author_channel_id in subscriber_channel_ids

                    # Add the author's channel ID to the set of subscriber channel IDs
                    subscriber_channel_ids.add(author_channel_id)

                    comments.append([
                        author_name,
                        comment['publishedAt'],
                        like_count_per_comment,
                        comment['textDisplay'],
                        is_subscriber,
                        author_channel_id
                    ])

                next_page_token = response.get('nextPageToken')
                if not next_page_token or total_results >= max_results:
                    break

            df = pd.DataFrame(comments, columns=['author', 'published_at', 'like_count_per_comment', 'text', 'is_subscriber', 'authorChannelId'])

            # Fetch author subscriber counts
            author_subscribers = {}

            for channel_id in author_channel_ids:
                channel_response = youtube.channels().list(
                    part="statistics",
                    id=channel_id
                ).execute()

                subscriber_count = channel_response['items'][0]['statistics']['subscriberCount']
                author_subscribers[channel_id] = subscriber_count

            # Add subscriber count column
            df['author_subscribers'] = df['authorChannelId'].map(author_subscribers)

            df['Word_Count'] = df['text'].apply(lambda text: len(str(text).split()) if pd.notnull(text) else 0)

            # Fetch video details (video_id, video_title, video_published_at, total_likes, total_dislikes, total_comments, total_views)
            video_request = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            )
            video_response = video_request.execute()

            video_item = video_response['items'][0]
            video_title = video_item['snippet']['title']
            video_published_at = video_item['snippet']['publishedAt']
            total_likes = video_item['statistics']['likeCount']
            total_comments = video_item['statistics']['commentCount']
            total_views = video_item['statistics']['viewCount']

            # Fetch channel statistics (number of subscribers)
            channel_id = video_item['snippet']['channelId']
            channel_request = youtube.channels().list(
                part="statistics",
                id=channel_id
            )
            channel_response = channel_request.execute()
            channel_item = channel_response['items'][0]
            total_subscribers = channel_item['statistics']['subscriberCount']

            # Convert total likes, total views, and total comments to k(1000's) format
            total_likes_int = int(total_likes)
            total_views_int = int(total_views)
            total_comments_int = int(total_comments)

            total_likes_k = f"{total_likes_int / 1000:.1f}k"
            total_views_k = f"{total_views_int / 1000:.1f}k"
            total_comments_k = total_comments_int
            # Create rectangle color boxes with the information
            likes_box = f"""
                <div style='background-color: green; color: white; padding: 4px; text-align: center; width: 250px; margin: auto;'>
                    <p style='font-size: 24px; font-weight: bold;'>Total Likes</p>
                    <p style='font-size: 40px; font-weight: bold;'>{total_likes_k}</p>
                </div>
            """
            views_box = f"""
                <div style='background-color: blue; color: white; padding: 4px; text-align: center; width: 250px; margin: auto;'>
                    <p style='font-size: 24px; font-weight: bold;'>Total Views</p>
                    <p style='font-size: 40px; font-weight: bold;'>{total_views_k}</p>
                </div>
            """
            comments_box = f"""
                <div style='background-color: orange; color: white; padding: 4px; text-align: center; width: 250px; margin: auto;'>
                    <p style='font-size: 24px; font-weight: bold;'>Total Comments</p>
                    <p style='font-size: 40px; font-weight: bold;'>{total_comments_k}</p>
                </div>
            """

            # Display the boxes with the information in a grid layout
            st.markdown("<h3 style='text-align: center; font-weight: bold;'>Video Information</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; font-size: 28px; font-weight: bold;'>{video_title}</p>",
                        unsafe_allow_html=True)

            # Use the grid layout to display the boxes side by side
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(likes_box, unsafe_allow_html=True)
            with col2:
                st.markdown(comments_box, unsafe_allow_html=True)
            with col3:
                st.markdown(views_box, unsafe_allow_html=True)

            st.markdown("<br><br>", unsafe_allow_html=True)

            df['processed_text'] = df['text'].apply(text_processing)

            # Perform sentiment analysis on the 'processed_text' column
            df['sentiment'] = df['processed_text'].apply(lambda x: sentiment_score(x))

            # Convert polarity scores to sentiment labels
            df['sentiment_class'] = df['sentiment'].apply(
                lambda x: 'Positive' if x > 0 else 'Neutral' if x == 0 else 'Negative')
            sentiment_mapping = {
                'Positive': 2,
                'Neutral': 1,
                'Negative': 0
            }

            # Map the sentiment labels to integer values using the mapping dictionary
            df['sentiment_class'] = df['sentiment_class'].map(sentiment_mapping)
            # Split the DataFrame into positive and negative comments

            positive_df = df[df['sentiment_class'] == 2]
            negative_df = df[df['sentiment_class'] == 0]

            col1, col2 = st.columns(2)
            # Display positive comments table in the first column
            with col1:
                st.markdown("<h3 style='text-align: center; color: green;'>Positive Comments</h3>", unsafe_allow_html=True)
                positive_table = positive_df[['author', 'like_count_per_comment', 'text']]
                st.dataframe(positive_table)

            # Display negative comments table in the second column
            with col2:
                st.markdown("<h3 style='text-align: center; color: red;'>Negative Comments</h3>", unsafe_allow_html=True)
                negative_table = negative_df[['author', 'like_count_per_comment', 'text']]
                st.dataframe(negative_table)

            positive_comments = " ".join(df[df['sentiment_class'] == 1]['processed_text'])
            negative_comments = " ".join(df[df['sentiment_class'] == 0]['processed_text'])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("<h3 style='text-align: center; color: green;'>Positive Comments Word Cloud</h3>",
                            unsafe_allow_html=True)
                positive_comments_wordcloud = create_word_cloud(positive_comments)
                st.pyplot(positive_comments_wordcloud)

            # Create word cloud for negative comments
            with col2:
                st.markdown("<h3 style='text-align: center; color: red;'>Negative Comments Word Cloud</h3>",
                            unsafe_allow_html=True)
                negative_comments_wordcloud = create_word_cloud(negative_comments)
                st.pyplot(negative_comments_wordcloud)

            most_liked_comments = df.sort_values(by='like_count_per_comment', ascending=False)[['author', 'like_count_per_comment', 'text']]
            # Display the most liked comments table
            st.markdown("<h3 style='text-align: center;'>Most Liked Comments</h3>", unsafe_allow_html=True)
            st.table(most_liked_comments.head(5))

            # Calculate the count of comments in each sentiment category
            sentiment_counts = df['sentiment_class'].value_counts()

            # Create a bar plot to visualize the sentiment distribution
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.bar(sentiment_counts.index, sentiment_counts.values, color=['green', 'red', 'gray'])
            ax1.set_xticks(sentiment_counts.index)
            ax1.set_xticklabels(['Positive', 'Negative', 'Neutral'])
            ax1.set_ylabel('Count of Comments')
            ax1.set_xlabel('Sentiment Category')

            df['published_at'] = pd.to_datetime(df['published_at'])

            # Group by date and calculate average sentiment for each date
            sentiment_over_time = df.groupby(pd.Grouper(key='published_at', freq='W'))['sentiment'].mean().reset_index()

            # Create a line plot for sentiment trend over time on a weekly basis
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(sentiment_over_time['published_at'], sentiment_over_time['sentiment'], marker='o')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Average Sentiment Score')
            ax2.grid(True)
            plt.tight_layout()

            # Display both plots side by side using Streamlit columns
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown("<h3 style='text-align: center;'>Sentiment Distribution of YouTube Comments</h3>", unsafe_allow_html=True)
                st.pyplot(fig1)

            with col2:
                st.markdown("<h3 style='text-align: center;'>Sentiment Trend Over Time</h3>", unsafe_allow_html=True)
                st.pyplot(fig2)

            # Define weights for each factor
            weight_like_count_per_comment = 0.5
            weight_sentiment_score = 0.3
            weight_author_subscribers = 0.7
            weight_word_count = 0.2

            # Normalize numerical columns
            scaler = MinMaxScaler()
            df['like_count_per_comment_normalized'] = scaler.fit_transform(df[['like_count_per_comment']])
            df['author_subscribers_normalized'] = scaler.fit_transform(df[['author_subscribers']])
            df['Word_Count_normalized'] = scaler.fit_transform(df[['Word_Count']])

            # Calculate the score for each comment
            df['score'] = (
                    weight_like_count_per_comment * df['like_count_per_comment_normalized'] +
                    weight_sentiment_score * df['sentiment'] +
                    weight_author_subscribers * df['author_subscribers_normalized'] +
                    weight_word_count * df['Word_Count_normalized']
            )

            # Filter the comments with score > 30%
            top_comments_to_answer = df[df['score'] > 0.3]  # Adjust the threshold (0.3) as needed

            # Sort the comments based on the score in descending order
            top_comments_to_answer = top_comments_to_answer.sort_values(by='score', ascending=False)

            # Display the top comments to answer
            st.markdown("<h3 style='text-align: center;'>Top Comments to Answer</h3>",
                        unsafe_allow_html=True)
            top_comments_to_answer.index = range(1, len(top_comments_to_answer) + 1)
            st.table(top_comments_to_answer[['author', 'text', 'like_count_per_comment', 'score']])

        else:
            st.markdown("<h3 style='text-align: center; color: red;'>Invalid YouTube URL. Please enter a valid URL.</h3>",
                        unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

with st.container():
    st.markdown("<h2 style='text-align: center;'>About Project</h2>", unsafe_allow_html=True)
    st.markdown("This is a YouTube Comments Analyzer project that empowers content creators with data visualization and actionable feedback. "
                "It uses OpenAI's GPT-3.5 Turbo model to generate engaging responses to comments for video creators. "
                "The user can paste the YouTube video URL and get sentiment analysis of comments, word clouds, and actionable feedback "
                "to engage with their audience. The project is powered by Streamlit and YouTube API for fetching video comments and details.")

    st.markdown("<h3 style='text-align: center;'>How to Use</h3>", unsafe_allow_html=True)
    st.markdown("1. Paste the YouTube video URL in the text box and select the number of comments to analyze.")
    st.markdown("2. Click 'Analyze Comments' to fetch comments and sentiment analysis.")
    st.markdown("3. Explore the sentiment distribution, word clouds, and most liked comments.")
    st.markdown("4. Find the top comments to answer and improve your content based on feedback.")
    st.markdown("5. Use the AI-generated replies to engage with your audience more effectively on ResponseBot page.", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Rest of your Home page content here (e.g., sentiment distribution plot, word clouds, etc.)

# Add a horizontal line to separate the About Project from the About Author section
st.markdown("<hr>", unsafe_allow_html=True)




