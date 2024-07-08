import streamlit as st
from src.pipeline.file_preprocessor import FilePreprocesser
from src.components.details_fetcher import DetailsFetcher
import matplotlib.pyplot as plt
import seaborn as sns
from src.exceptions import CustomException
import sys
import warnings
warnings.simplefilter("ignore")

st.sidebar.title("Whatsapp Chat Analyzer")

file_preprocessor_obj=FilePreprocesser()
details_fetcher_obj=DetailsFetcher()

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = file_preprocessor_obj.process(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    try:
        user_list.remove('group_notification')
    except Exception as e:
        CustomException(e,sys)
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        # Stats Area
        num_messages, words, num_media_messages, num_links = details_fetcher_obj.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)


        st.title("Sentiment Analysis")
        col1,col2=st.columns(2)
        with col1:
            st.header("Daily Sentiment Analysis")
            sentiment_timeline=details_fetcher_obj.daily_sentiment_timeline(selected_user,df)
            fig, ax = plt.subplots()
            ax.plot(sentiment_timeline['only_date'], sentiment_timeline['sentiment'], color='green')

            # Customize ticks and rotation
            plt.xticks(rotation='vertical')
            plt.yticks([2, 1, 0])

            # Display the plot in Streamlit
            st.pyplot(fig)

        with col2:
            st.header("Monthly Sentiment Analysis")
            timeline = details_fetcher_obj.monthly_sentiment_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['sentiment'], color='green')
            plt.xticks(rotation='vertical')
            plt.yticks([2, 1, 0])

            st.pyplot(fig)
        # monthly timeline
        st.title("Monthly Timeline")
        timeline = details_fetcher_obj.monthly_timeline(selected_user,df)
        fig,ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'],color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = details_fetcher_obj.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # activity map
        st.title('Activity Map')
        col1,col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = details_fetcher_obj.week_activity_map(selected_user,df)
            fig,ax = plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = details_fetcher_obj.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = details_fetcher_obj.activity_heatmap(selected_user,df)
        fig,ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x,new_df = details_fetcher_obj.most_busy_users(df)
            fig, ax = plt.subplots()

            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # WordCloud
        st.title("Wordcloud")
        df_wc = details_fetcher_obj.create_wordcloud(selected_user,df)
        fig,ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        most_common_df = details_fetcher_obj.most_common_words(selected_user,df)

        fig,ax = plt.subplots()

        ax.barh(most_common_df[0],most_common_df[1])
        plt.xticks(rotation='vertical')

        st.title('Most commmon words')
        st.pyplot(fig)

        # emoji analysis
        emoji_df = details_fetcher_obj.emoji_helper(selected_user,df)
        st.title("Emoji Analysis")

        col1,col2 = st.columns(2)

        with col1:
            st.dataframe(emoji_df)
        with col2:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f")
            plt.show()











