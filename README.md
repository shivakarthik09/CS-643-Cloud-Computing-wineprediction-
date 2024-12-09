# CS-643-Cloud-Computing-wineprediction-


    <h2>Objective</h2>
    <p>This assignment focuses on developing parallel machine learning applications on the Amazon AWS cloud platform. The primary goal is to learn:</p>
    <ul>
        <li>Utilizing Apache Spark to train a machine learning model in parallel across multiple EC2 instances.</li>
        <li>Employing Spark’s MLlib for machine learning model development and application in the cloud.</li>
        <li>Leveraging Docker to containerize the ML model, simplifying deployment across environments.</li>
    </ul>

    <h2>Introduction</h2>
    <p>The task involves building and deploying a wine quality prediction model. This model will be trained using Spark on AWS, utilizing multiple EC2 instances for parallel processing. The implementation language will be Java on an Ubuntu Linux environment.</p>

    <h2>Methodology</h2>

    <h3>Dataset Description</h3>
    <ul>
        <li><strong>TrainingDataset.csv</strong>: Used for training the model on multiple EC2 instances in parallel.</li>
        <li><strong>ValidationDataset.csv</strong>: Employed for model validation and parameter tuning to optimize performance.</li>
    </ul>

    <h3>Model Development</h3>
    <ul>
        <li>The model is trained using Spark’s MLlib, starting with a basic logistic regression model, potentially exploring other models for enhanced performance.</li>
        <li>The model uses the training dataset for learning and the validation dataset for performance tuning.</li>
        <li>Classification approach considers wine scores from 1 to 10, allowing a multi-class classification model.</li>
    </ul>
</body>
</html>
