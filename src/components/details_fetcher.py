import pandas as pd
from src.logger import logging
from src.exceptions import CustomException
import sys
from urlextract import URLExtract
from wordcloud import WordCloud
from collections import Counter
import emoji
class DetailsFetcher:
    def __init__(self):
        pass

    def fetch_stats(self,selected_user, df):

        extract = URLExtract()

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        num_messages = df.shape[0]

        words = []
        for message in df['message']:
            words.extend(message.split())

        num_media_messages = df[df['message'] == '<Media omitted>\n'].shape[0]

        links = []
        for message in df['message']:
            links.extend(extract.find_urls(message))

        return num_messages, len(words), num_media_messages, len(links)

    def most_busy_users(self,df):
        active_user = df['user'].value_counts().head()
        df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(
            columns={'index': 'name', 'user': 'percent'})
        return active_user, df

    def create_wordcloud(self,selected_user, df):

        f = open('data/stop_hinglish.txt', 'r')
        stop_words = f.read()

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        user_messages = df[(df['user'] != 'group_notification') & (df['message'] != '<Media omitted>\n')]

        def remove_stop_words(message):
            y = []
            for word in message.lower().split():
                if word not in stop_words:
                    y.append(word)
            return " ".join(y)

        wc = WordCloud(width=500, height=500, min_font_size=10, background_color='white')
        user_messages['message'] = user_messages['message'].apply(remove_stop_words)
        df_wordcloud = wc.generate(user_messages['message'].str.cat(sep=" "))
        return df_wordcloud

    def most_common_words(self,selected_user, df):

        f = open('data/stop_hinglish.txt', 'r')
        stop_words = f.read()

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        user_message = df[df['user'] != 'group_notification']
        user_message = user_message[user_message['message'] != '<Media omitted>\n']

        words = []

        for message in user_message['message']:
            for word in message.lower().split():
                if word not in stop_words:
                    words.append(word)

        most_common_df = pd.DataFrame(Counter(words).most_common(20))
        return most_common_df

    def emoji_helper(self,selected_user, df):
        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        emojis = []
        for message in df['message']:
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

        emoji_count = Counter(emojis)
        emoji_df = pd.DataFrame(emoji_count.most_common(), columns=['emoji', 'count'])

        return emoji_df

    def monthly_timeline(self,selected_user, df):

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

        time = []
        for i in range(timeline.shape[0]):
            time.append(timeline['month'][i] + "-" + str(timeline['year'][i]))

        timeline['time'] = time

        return timeline

    def daily_timeline(self,selected_user, df):

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        daily_timeline = df.groupby('only_date').count()['message'].reset_index()

        return daily_timeline

    def week_activity_map(self,selected_user, df):

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        return df['day_name'].value_counts()

    def month_activity_map(self,selected_user, df):

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        return df['month'].value_counts()

    def activity_heatmap(self,selected_user, df):

        if selected_user != 'Overall':
            df = df[df['user'] == selected_user]

        user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

        return user_heatmap