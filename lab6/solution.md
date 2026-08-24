# Lab 6 – Unsupervised Learning

## 1. What is the purpose of the neighborhood function in the SOM? How does it change the learning?

The neighborhood function in SOM decides which neurons should learn when an input is given. First, SOM finds the neuron that is closest to the input, which is called the winning neuron or BMU. But instead of updating only that neuron, it also updates the neurons around it.

At the beginning, the neighborhood is usually bigger, so more neurons learn together. As training continues, the neighborhood gets smaller, so the SOM can make more detailed adjustments. This is important because it helps similar data end up close to each other on the map.

Without the neighborhood function, SOM would act more like regular competitive learning where only the winner gets updated.

## 2. Intruder Detection Using SOM

For this system, I would use each login session as one input to the SOM. The input could include:

- Login time
- How long the person stayed logged in
- What programs they used
- How many programs they ran

Before training, I would preprocess the data because these features are very different from each other. For example, login time should be handled carefully because 11:59 PM and 12:01 AM are actually very close even though their numerical values look far apart. I could use sine and cosine values for time.

I would also normalize session length and number of programs. Program types could be represented using binary values, such as whether the user opened a browser, terminal, email program, and so on.

I would use a large amount of normal historical data because people's behavior can change depending on the day and time. Probably thousands of login sessions would be a good starting point. For the SOM size, something like a 10×10 map could work initially, and then I could increase it if the data has many different patterns.

After training, the SOM should create different clusters for normal user behaviors. If a new login is very different from those normal clusters, the system could flag it as suspicious.

I think this could help with intruder detection, but I would not depend on it alone. A normal user can sometimes behave differently, which could cause a false alarm. Also, an intruder who behaves similarly to the real user might not be detected.

## 3. Competitive Learning for Credit Card Fraud Detection

For the credit card problem, I would first turn each transaction into features such as:

- Transaction amount
- Type of shop
- Time of day
- Day of the week

The values should be normalized so that something like transaction amount does not have too much influence just because its numbers are larger.

Then I could use SOM or another competitive learning method to learn the normal spending patterns of each customer. For example, one person might usually spend $10–$50 at grocery stores and restaurants, while another person might regularly make larger purchases.

The model could create clusters representing these different spending behaviors.

When a new transaction happens, I could compare it with the customer's normal clusters. If it is very far from their usual pattern, it could be considered suspicious. It would be even more suspicious if several unusual transactions suddenly happened after each other.

I think this could work fairly well for finding obvious changes, but it would not be perfect. For example, if someone goes on vacation and starts spending money in a different city, the model might think the card was stolen. On the other hand, if a thief spends money in a way that looks normal for that customer, the model might not notice.

Another issue is that there are way more normal transactions than stolen-card transactions. Because of this, the model can become very good at representing normal behavior but not learn much from the small number of fraud examples.

One way to handle this is to mainly train the SOM on normal transactions and treat transactions that are far away from normal clusters as possible fraud. The known stolen-card transactions can then be used to test the system and decide how unusual a transaction needs to be before it is flagged.

## 4. K-Means Clustering by Hand

First I worked out the initial cluster means. To do this I checked the distance between every pair of points and found that Subject 1 (1.0, 1.0) and Subject 4 (5.0, 7.0) are the furthest apart, with a distance of about 7.21. So these become my starting cluster means:

- Cluster 1 mean = (1.0, 1.0)
- Cluster 2 mean = (5.0, 7.0)

**Iteration 1**

I worked out the distance from every subject to both means and assigned each subject to whichever mean it was closer to.

- Subjects 1, 2, and 3 ended up closer to Cluster 1
- Subjects 4, 5, 6, and 7 ended up closer to Cluster 2

(Subject 3 was actually equal distance to both, so I just left it in Cluster 1.)

New means after this round:
- Cluster 1 = (1.83, 2.33)
- Cluster 2 = (4.13, 5.38)

**Iteration 2**

I recalculated the distances using the new means. This time Subject 3 switched over to Cluster 2 because it was now closer to that mean.

- Cluster 1 = Subjects 1, 2
- Cluster 2 = Subjects 3, 4, 5, 6, 7

New means:
- Cluster 1 = (1.25, 1.50)
- Cluster 2 = (3.90, 5.10)

**Iteration 3**

I checked again with the updated means, and this time nobody changed cluster. Since the assignments were the same as the previous round, the algorithm has converged and I can stop here.

**Final answer:**

- **Cluster 1** = {Subject 1, Subject 2}, mean = (1.25, 1.50)
- **Cluster 2** = {Subject 3, Subject 4, Subject 5, Subject 6, Subject 7}, mean = (3.90, 5.10)

Basically, the two lowest-scoring subjects grouped together, and everyone else with noticeably higher scores on both variables grouped into the second cluster. It only took two rounds of reassigning before the clusters settled down and stopped changing.