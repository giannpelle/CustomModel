using System;
using Microsoft.ML.Data;

public class AsiQuestion
{
    [LoadColumn(0), ColumnName("Category")]
    public string Category { get; set; }
    [LoadColumn(1), ColumnName("Question")]
    public string Question { get; set; }
}

public class QuestionPrediction
{
    [ColumnName("PredictedLabel")]
    public string Category;
    [ColumnName("Score")]
    public Single[] Score;
}
