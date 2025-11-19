import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Dataset .csv")

st.title("Restaurant Data Analysis Dashboard")


# LEVEL 1
st.header("LEVEL 1")

# Task 1
st.subheader("Task 1: Cuisine Analysis")

# Top 3 cuisines
data_top = data["Cuisines"].value_counts()
st.write("Top 3 Most Common Cuisines:")
st.table(data_top.head(3))

# Percentage of restaurants serving each cuisine
data_per = (data_top / data_top.sum()) * 100
data_per = data_per.reset_index()
data_per.columns = ["Cuisines", "Percentage"]
st.write("Percentage of restaurants that serve each cuisine:")
st.table(data_per.head(3))


# Task 2
st.subheader("Task 2: City Analysis")

# City with highest restaurants
data_city = data["City"].value_counts()
st.write("City with Highest Number of Restaurants:")
st.table(data_city.head(1))

# Average rating per city
data_rating = data.groupby("City")["Aggregate rating"].mean()
st.write("Average Rating of Restaurants by City:")
st.dataframe(data_rating)

# City with highest avg rating
st.write("City with Highest Average Rating:")
st.table(data_rating.sort_values(ascending=False).head(1))


# Task 3
st.subheader("Task 3: Price Range Distribution")

fig, ax = plt.subplots()
ax.hist(data["Price range"])
ax.set_xlabel("Price Range")
ax.set_ylabel("Frequency")
ax.set_title("Price Range Distribution")
st.pyplot(fig)

data_price = data["Price range"].value_counts(normalize=True) * 100
data_price = data_price.reset_index()
data_price.columns = ["Price Range", "Percentage"]
st.write("Percentage of restaurants in each price range:")
st.table(data_price)


# Task 4
st.subheader("Task 4: Online Delivery Availability")

data_delivery = data["Has Online delivery"].value_counts(normalize=True) * 100
data_delivery = data_delivery.reset_index()
data_delivery.columns = ["Has Online delivery", "Percentage"]
st.table(data_delivery)

delivery_rating = data.groupby("Has Online delivery")["Aggregate rating"].mean()
st.write("Average Rating Comparison (Online Delivery vs No Delivery):")
st.table(delivery_rating)


# LEVEL 2
st.header("LEVEL 2")

# Task 1
st.subheader("Task 1: Ratings Distribution")

fig, ax = plt.subplots()
ax.hist(data["Aggregate rating"], bins=10, edgecolor="black")
ax.set_xlabel("Aggregate Rating")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Restaurant Ratings")
st.pyplot(fig)

common_rating = data["Aggregate rating"].value_counts()
st.write("Most Common Ratings:")
st.table(common_rating.head())

avg_votes = data["Votes"].mean()
st.write(f"Average number of votes received by restaurants: {avg_votes:.2f}")


# Task 2
st.subheader("Task 2: Cuisine Combinations")

data_cuisine = data["Cuisines"].value_counts()
st.write("Most common cuisine combinations:")
st.table(data_cuisine.head())

data_cuisine_rating = data.groupby("Cuisines")["Aggregate rating"].mean()
st.write("Cuisine combinations with highest ratings:")
st.table(data_cuisine_rating.sort_values(ascending=False).head())


# Task 3
st.subheader("Task 3: Geographic Analysis")

fig, ax = plt.subplots()
ax.scatter(data["Longitude"], data["Latitude"], alpha=0.5)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographic Distribution of Restaurants")
st.pyplot(fig)

city_counts = data["City"].value_counts().head(5)
st.write("Top 5 Cities with Most Restaurants:")
st.table(city_counts)


# Task 4
st.subheader("Task 4: Restaurant Chains")

restaurant_chains = data["Restaurant Name"].value_counts()
st.write("Top Restaurant Chains:")
st.table(restaurant_chains.head())

chain_ratings = (
    data.groupby("Restaurant Name")["Aggregate rating"]
    .agg(["count", "mean"])
    .sort_values("count", ascending=False)
)
st.write("Chain Analysis (Ratings & Popularity):")
st.table(chain_ratings.head())


# LEVEL 3
st.header("LEVEL 3")

# Task 1
st.subheader("Task 1: Review Analysis")

text_review = data["Rating text"].value_counts()
st.write("Review Sentiment Categories:")
st.table(text_review)

data["Rating text Length"] = data["Rating text"].str.len()
avg_length = data["Rating text Length"].mean()
st.write(f"Average length of reviews: {avg_length:.2f} characters")

positive_words = ["Excellent", "Very Good", "Good"]
negative_words = ["Average", "Poor"]

pos_count = data["Rating text"].isin(positive_words).sum()
neg_count = data["Rating text"].isin(negative_words).sum()

st.write(f"Positive reviews: {pos_count}")
st.write(f"Negative reviews: {neg_count}")


# Task 2
st.subheader("Task 2: Votes & Ratings Relationship")

restaurants_votes = data.sort_values(by="Votes", ascending=False)
st.write("Restaurants with Highest Votes:")
st.table(
    restaurants_votes[["Restaurant Name", "City", "Aggregate rating", "Votes"]].head(2)
)

st.write("Restaurants with Lowest Votes:")
st.table(
    restaurants_votes[["Restaurant Name", "City", "Aggregate rating", "Votes"]].tail(2)
)

corr_votes_rating = data["Votes"].corr(data["Aggregate rating"])
st.write(f"Correlation between Votes and Rating: {corr_votes_rating:.2f}")


# Task 3
st.subheader("Task 3: Price Range vs Delivery & Booking")

data["Has Online delivery_numeric"] = data["Has Online delivery"].apply(
    lambda x: 1 if x == "Yes" else 0
)
data["Has Table booking_numeric"] = data["Has Table booking"].apply(
    lambda x: 1 if x == "Yes" else 0
)

relation_price_delivery = (
    data.groupby("Price range")[
        ["Has Online delivery_numeric", "Has Table booking_numeric"]
    ].mean()
    * 100
)
st.write("Service Availability by Price Range (%):")
st.table(relation_price_delivery)

fig, ax = plt.subplots()
relation_price_delivery.plot(kind="bar", ax=ax)
ax.set_ylabel("Percentage (%)")
ax.set_title("Online Delivery & Table Booking by Price Range")
st.pyplot(fig)
