from pyspark.sql import SparkSession

spark = SparkSession.builder \
                    .master('local[*]') \
                    .appName('mySparkSession') \
                    .getOrCreate()

# What version of Spark?
print('Running Spark version '+spark.version)

# 1. Read csv file
passengers_train = spark.read.csv('data/train.csv', header=True, inferSchema=True, nullValue='NA')
passengers_test = spark.read.csv('data/test.csv', header=True, inferSchema=True, nullValue='NA')
print('Size of train set: %d' % passengers_train.count())
print('Size of test set: %d' % passengers_test.count())
passengers_train.show(5)
passengers_train.printSchema()
passengers_train.dtypes

# Number of records with missing 'Age' values
print('Null values in Age:')
print(passengers_train.filter('Age IS NULL').count())
print(passengers_test.filter('Age IS NULL').count())

# Number of records with missing 'Cabin' values
print('Null value in Cabin:')
print(passengers_train.filter('Cabin IS NULL').count())
print(passengers_test.filter('Cabin IS NULL').count())

# Number of records with missing 'Embarked' values
print('Null value in Embarked:')
print(passengers_train.filter('Embarked IS NULL').count())
print(passengers_test.filter('Embarked IS NULL').count())
passengers_train = passengers_train.filter('Embarked IS NOT NULL')
print(passengers_train.filter('Embarked IS NULL').count())
print(passengers_test.filter('Embarked IS NULL').count())

# 2. Feature engineering

# 2.1 Regex to find personal titles of passengers
# Option (a): Engineer the feature in the original data frame
# See: https://stackoverflow.com/questions/46410887/pyspark-string-matching-to-create-new-column
from pyspark.sql.functions import regexp_extract, col

"""
# Here regex('(.)(by)(\s+)(\w+)') means
# (.) - Any character (except newline)
# (,) - A comma in the text
# (\s+) - One or many spaces
# (\w+) - Alphanumeric or underscore chars of length one or more
# and group_number is 4 because group (\w+) is in 4th position in expression we want to extract
passengers_train = passengers_train.withColumn('Title', regexp_extract(col('Name'), '(.)(,)(\s+)(\w+)', 4))
passengers_test = passengers_test.withColumn('Title', regexp_extract(col('Name'), '(.)(,)(\s+)(\w+)', 4))
print(passengers_train.filter('Title IS NULL').count())
print(passengers_test.filter('Title IS NULL').count())
"""

# Option (b): Create a Transformer so that feature creation can be included in Pipeline
# A Transformer is an abstraction that includes feature transformers and learned models. 
# Technically, a Transformer implements a method transform(), which converts one DataFrame into another,
# generally by appending one or more columns (https://spark.apache.org/docs/latest/ml-pipeline.html.)
# See: https://towardsdatascience.com/pyspark-wrap-your-feature-engineering-in-a-pipeline-ee63bdb913
from pyspark.ml.pipeline import Transformer

class TitleExtractor(Transformer):
    def __init__(self, inputCol='Name', outputCol='Title'):
        self.inputCol = inputCol
        self.outputCol = outputCol
        
    def this():
        this(Identifiable.randomUID("titleextractor"))
    def copy(extra):
        defaultCopy(extra)
                   
    def _transform(self, df):
        return df.withColumn('Title', regexp_extract(col('Name'), '(.)(,)(\s+)(\w+)', 4))

title_extractor = TitleExtractor(inputCol='Name')

# 2.2 String indexing and One-hot encoding
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator 
indexer1 = StringIndexer(inputCol='Sex', outputCol='Sex_idx')
indexer2 = StringIndexer(inputCol='Embarked', outputCol='Embarked_idx')
indexer3 = StringIndexer(inputCol='Title', outputCol='Title_idx')
onehot = OneHotEncoderEstimator(inputCols=['Sex_idx','Embarked_idx', 'Title_idx'], \
                                outputCols=['Sex_dummy', 'Embarked_dummy', 'Title_dummy'])

# 2.3 Vector assembler
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['Pclass', 'Fare', 'Sex_dummy', 'Embarked_dummy', 'Title_dummy'], \
                            outputCol='features')

# 3. Models
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
tree = DecisionTreeClassifier(labelCol='Survived')
rf = RandomForestClassifier(labelCol='Survived')

# 4. Create pipeline
from pyspark.ml import Pipeline
#pipeline = Pipeline(stages=[indexer, onehot, assembler, tree])
pipeline = Pipeline(stages=[title_extractor, indexer1, indexer2, indexer3, onehot, assembler, rf])

# 5. Fit the model
pipeline = pipeline.fit(passengers_train)

# 6. Make predictions
#from pyspark.ml.evaluation import BinaryClassEvaluator
prediction = pipeline.transform(passengers_train)
prediction.show(5)
prediction.select('Survived', 'prediction', 'probability').show(5, False)

# Create a confusion matrix
prediction.groupBy('Survived', 'prediction').count().show()

TP = prediction.filter('Survived == 1 AND prediction == 1').count()
TN = prediction.filter('Survived == 0 AND prediction == 0').count()
FP = prediction.filter('Survived == 0 AND prediction == 1').count()
FN = prediction.filter('Survived == 1 AND prediction == 0').count()

# Compute accuracy
accuracy = (TP+TN)/(TP+TN+FP+FN)
print('Accuracy is %f' % accuracy)
