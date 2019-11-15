# Data-Analytics-for-Cyber-Security-for-tweets-Email-Classification
# Project Description:  
The popularity of social media networks, such as Twitter, leads to an increasing number of spamming activities. Researchers employed various machine learning methods to detect Twitter spams. In this assignment, you are required to classify spam tweets by using provided datasets. In given train , test 1 and test2 data we have these columns - "account_age"        "no_followers" , "no_folowing", "no_userfavirate", "No_lists"  ,"no_tweets", "no_retweets", "no_tweetsfaviorate", "no_hashtag", "no_usermention", "no_urls"  , "no_char", "no_digits","spam". Data set contain train has 1998 rows and 14 columns and test1 and test two both has 2000 rows and 14 variables. By this data set we have to classify the Tweets mails that are spam or non-spammer. Most businesses, government organizations, and academic institutions house dedicated network security teams which help defend the integrity of internal data or client-facing sites against advanced threats or intrusion attacks. Even at home, you may have a firewall to protect your personal network from outside hackers or attacks. However, as we see on the news quite often, even the most sophisticated organizations and individuals often will have their reputable websites or twitter account passwords compromised.
Recognizing this challenge, our project focused on developing machine learning (ML) algorithms to tackle detecting tweets spams. We also placed emphasis on balancing model accuracy with interpretability and computational resources at our disposal.

What is a cyber-attack?
Cyber-attacks are malicious Internet operations launched mostly by criminal organizations whose goal may be to steal money, financial data, intellectual property, or to simply disrupt the operations of a certain company. Countries also get involved in so-called state-sponsored cyber-attacks, where they seek to learn classified information on a geopolitical rival, or simply to “send a message.” 
The global cost of cybercrime for 2015 was $500 billion (BOLD NUMBER).That’s more than 5 times Google’s yearly cash flow of 90 billion dollars. And that number is set to grow tremendously, to around 2 trillion dollars by 2019. In this article we want to explore the types of attacks used by cybercriminals to drive up such a huge figure and help you understand how they work and affect you.

# Objectives
• To apply skills and knowledge acquired throughout the semester in classification
algorithms and machine learning process.
• To rationalize the use of machine learning algorithms to effectively and efficiently process
of data in big size.
• To demonstrate ability to use R to perform email classification tasks that are common for
corporate security analyst.
• To scientifically conduct and document machine learning experiments for analytics
purposes.
# Performance Comparison: 
The algorithms used in the study consist of Logistic Regression, Radom Forest, Gradient Boosting, Support Vector Classifier and Neural Network. Without oversampling and parameter optimization, all algorithms show around 90% accuracy overall but about 20% sensitivity on fitting and prediction of tweets spam.  After applying cross validation, all the algorithms except the Support Vector Classifier showed improvement on balanced sensitivities and the top two improvements were from Gradient Boosting and Neural Network. The following outputs are then generated from Logistic Regression, Gradient Boosting and Neural Network. Here Logistic Regression is served as a baseline other algorithm to be compared with.
Compared with those true values, the prediction output on test set shows:
1. The baseline model is accurate in making predictions in general: It is able to predict correctly over 90% of the time. However, it can only predict tweets spam correctly 23% of the time. So the model is not informative enough to our project goal.
2. Random forest has been improved on predicting tweets spam correctly to 66% of the time and it is able to filter out those who will not tweets spam 84% of the time.
3. Neural Network has been improved further on predicting tweets spam correctly to 84% of the time, however, its ability of filtering out those who will not spam was reduced to 43% of the time.
4.  Compared with random forest, Neural Network prediction is more expensive because of its failure to identify those who will not spam.


# CONCLUSION: 
Cyber-security represent a common and dangerous threat for every individual who is connected to the internet and all enterprises. As individual using an Android device, cautious downloads of apps can prevent spam activities from occurring on those devices. Increased cyber security training for users is highly recommended to counteract the threat of phishing emails. Ransomware has become a new and lucrative trend among the hacker community, prompting increased investments in cyber security across all industries. Neural Network has the best ability to predict spam tweets but gives no insight on feature influence.Unsupervised machine learning can be used to identify factors that cluster organizations into high-risk or low-risk groups, which in turn can provide helpful information to implement security measures that can not only detect cyber threats faster but also prevent recurring security. In the future, Deep Learning will be another powerful tool to detect unknown network intrusions.

# References
•	D.  Schuff,  O.  Turetke,  D.   Croson,  F  2007, ‘Managing  Email  Overload:  Solutions  and  Future Challenges’, IEEE Computer Society, vol. 40, No. 2, pp. 31-36

•	N. Kushmerick, T. Lau, 2005, ‘Automated Email Activity   Management:   An  Unsupervised   learning Approach’,   Proceedings   of   10th   International  Conference   on  Intelligent   User  Interfaces,   ACM Press, pp. 67-74

•	L. Zhou, E Hovy, 2005, “On the Summarization of   Dynamically   Introduced   Information:   Online Discussions  and  Blogs”,`In  Proceedings  of  AAAI-2006   Spring   Symposium   on   Computational
