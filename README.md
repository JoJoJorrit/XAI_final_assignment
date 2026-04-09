What is this project about?

When banks use AI to decide who gets a credit card, the process often feels like a black box. You get a "yes" or a "no," but you rarely get a clear reason why. This is a problem because people have a right to know how these decisions are made, especially under laws like the GDPR.

In this project, I used a machine learning model to predict credit defaults and then applied two different methods to explain the results. The goal was to see which method is more helpful for a regular person who just wants to know how to get approved in the future.
The methods I used

I compared two different ways of explaining the AI:

    SHAP: This is a mathematical way to see which factors were most important to the model. For example, it might show that your payment history mattered more than your age. It is great for experts, but can be a bit too technical for customers.

    DiCE (Counterfactuals): This is a newer approach that gives "what-if" examples. Instead of just giving you a score, it tells you exactly what to change—like "if you paid 500 dollars more on your last bill, you would have been approved."

How to use the code

To run this project on your own computer, you will need to install a few Python libraries first:

pip install pandas shap dice-ml scikit-learn matplotlib

Once you have those, you can run the main script:

python Final_assignment.py

Make sure the file named final_data.csv is in the same folder as the script, or it will not be able to find the data.
What is in this repository?

    Final_assignment.py: This is the main Python script. It trains the AI model and then runs the SHAP and DiCE explanations.

    final_data.csv: This is the dataset I used for the project.

    README.md: This file you are reading now.

    Final_Assignment_XAI-1.pdf: The report I wrote about this subject.

References

I used the following academic papers to help build this project:

    Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications.

    Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.

    Mothilal, R. K., Sharma, A., & Tan, C. (2020). Explaining machine learning classifiers through diverse counterfactual explanations. FAT*.
