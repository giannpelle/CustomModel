using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.LightGbm;

namespace CustomModel
{
    class Program
    {

        private static string _appPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string _trainDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "asi_train_set.csv");
        private static string _testDataPath => Path.Combine(_appPath, "..", "..", "..", "Data", "asi_test_set.csv");
        private static string _modelPath => Path.Combine(_appPath, "..", "..", "..", "Models", "asi_sdca_model.zip"); //bayes_model gbm_model sdca_model

        private static MLContext _mlContext;

        static void Main(string[] args)
        {
            _mlContext = new MLContext();

            CreateModel();

            Console.WriteLine("Finished");

        }

        // genera un nuovo modello ML 
        public static void CreateModel()
        {
            // carica i dati a partire da un file CSV
            var trainingDataView = _mlContext.Data.LoadFromTextFile<AsiQuestion>(_trainDataPath, separatorChar: ';', hasHeader: true);
            
            // crea una pipeline di trasformazioni da applicare ai dati prima di elaborare il modello
            var pipeline = CreateModelPipeline();

            // Allena il modello e lo salva su file
            TrainAndSaveModel(trainingDataView, pipeline);
        }

        // Test model and get metrics
        public static void EvaluateModel()

        {
            // carica il modello dal path dello zip
            var model = _mlContext.Model.Load(_modelPath, out var modelInputSchema);

            // valuta il modello
            Evaluate(model);
        }

        // extracts and transforms the data then returns the pipeline
        public static IEstimator<ITransformer> CreateModelPipeline()
        {
            //Sdca Trainer
            var sdcaPipeline = _mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Question", outputColumnName: "QuestionEncoded")
                .Append(_mlContext.Transforms.Concatenate("Features", "QuestionEncoded"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //One versus All Trainer
            var oneVsAllpipeline = _mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Question", outputColumnName: "QuestionEncoded")
                .Append(_mlContext.Transforms.Concatenate("Features", "QuestionEncoded"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.OneVersusAll(_mlContext.BinaryClassification.Trainers.SdcaLogisticRegression()))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //Naive Bayes Trainer VERY BADDDD
            var naiveBayesPipeline = _mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Question", outputColumnName: "QuestionEncoded")
                .Append(_mlContext.Transforms.Concatenate("Features", "QuestionEncoded"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.NaiveBayes("Label", "Features"))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            //Light Gbm Trainer
            var options = new LightGbmMulticlassTrainer.Options
            {
                Booster = new DartBooster.Options()
                {
                    TreeDropFraction = 0.15,
                    XgboostDartMode = false
                }
            };

            var lightGbmPipeline = _mlContext.Transforms.Text.FeaturizeText(inputColumnName: "Question", outputColumnName: "QuestionEncoded")
                .Append(_mlContext.Transforms.Concatenate("Features", "QuestionEncoded"))
                .Append(_mlContext.Transforms.Conversion.MapValueToKey(inputColumnName: "Category", outputColumnName: "Label"))
                .Append(_mlContext.MulticlassClassification.Trainers.LightGbm(options))
                .Append(_mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return sdcaPipeline;
            //return oneVsAllpipeline;
            //return naiveBayesPipeline;
            //return lightGbmPipeline;
        }

        //creates the training algorithm class, trains the model, predicts area from training data then returns model
        public static void TrainAndSaveModel(IDataView trainingDataView, IEstimator<ITransformer> pipeline)
        {
            var trainedModel = pipeline.Fit(trainingDataView);
            _mlContext.Model.Save(trainedModel, trainingDataView.Schema, _modelPath);
        }

        //loads test_set, creates the multiclass evaluator, evaluates the model and create metrics then display the metrics
        public static void Evaluate(ITransformer model)
        {
            // avendo a disposizione un dataset molto piccolo si è scelo di usare usarlo interamente per la creazione del modello
            // in futuro sarà possibile ritagliare una parte del dataset per effettuare i test del modello e ottenere le metriche di accuratezza
            /*var testDataView = _mlContext.Data.LoadFromTextFile<AsiQuestion>(_testDataPath, separatorChar: ',', hasHeader: true);
            var testMetrics = _mlContext.MulticlassClassification.Evaluate(model.Transform(testDataView));

            Console.WriteLine($"*************************************************************************************************************");
            Console.WriteLine($"*       Metrics for Multi-class Classification model - Test Data     ");
            Console.WriteLine($"*------------------------------------------------------------------------------------------------------------");
            Console.WriteLine($"*       MicroAccuracy:    {testMetrics.MicroAccuracy}");
            Console.WriteLine($"*       MacroAccuracy:    {testMetrics.MacroAccuracy}");
            Console.WriteLine($"*       LogLoss:          {testMetrics.LogLoss}");
            Console.WriteLine($"*       LogLossReduction: {testMetrics.LogLossReduction}");
            Console.WriteLine($"*************************************************************************************************************");
            */
        }

        // effettua una predizione e restituisce il risultato
        private static void MakePrediction()
        {
            ITransformer loadedModel = _mlContext.Model.Load(_modelPath, out var modelInputSchema);
            AsiQuestion singleQuestion = new AsiQuestion() { Question = "Sto avendo problemi con la valorizzazione di magazzino di fine anno, aiuto!" };
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<AsiQuestion, QuestionPrediction>(loadedModel);
            var prediction = predictionEngine.Predict(singleQuestion);

            var dict = ConvertScoresArrayIntoScoresDictionary(predictionEngine.OutputSchema, "Score", prediction.Score);

            Console.WriteLine($"=============== Single Prediction - Result: {prediction.Category} Accuracy: ===============");
        }

        // converte l'array di scores in un dizionario con chiave categoria e valore lo score
        private static Dictionary<string, float> ConvertScoresArrayIntoScoresDictionary(DataViewSchema schema, string name, float[] scores)
        {
            Dictionary<string, float> result = new Dictionary<string, float>();
            var column = schema.GetColumnOrNull(name);

            var slotNames = new VBuffer<ReadOnlyMemory<char>>();
            column.Value.GetSlotNames(ref slotNames);
            var names = new string[slotNames.Length];
            var num = 0;
            foreach (var denseValue in slotNames.DenseValues())
            {
                result.Add(denseValue.ToString(), scores[num++]);
            }

            return result.OrderByDescending(c => c.Value).ToDictionary(i => i.Key, i => i.Value);
        }

    }
}
