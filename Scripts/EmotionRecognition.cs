using Newtonsoft.Json.Linq;
using System.Collections.Generic;
using UnityEngine;

public class EmotionRecognition : MonoBehaviour
{
    [Header("Model Config")]
    public TextAsset configFile; // Drag config.json here
    public TextAsset thresholdsFile; // New file for thresholds

    private Dictionary<int, string> idToLabel;
    private Dictionary<string, float> emotionThresholds;

    void Start()
    {
        if (configFile == null || thresholdsFile == null)
        {
            Debug.LogError("Config file or thresholds file not assigned in the Inspector!");
            return;
        }

        LoadConfig();
        LoadThresholds();
    }

    private void LoadConfig()
    {
        var config = JObject.Parse(configFile.text);
        idToLabel = config["id2label"].ToObject<Dictionary<int, string>>();
        Debug.Log("Config loaded successfully.");
    }

    private void LoadThresholds()
    {
        var thresholdData = JObject.Parse(thresholdsFile.text);
        emotionThresholds = thresholdData.ToObject<Dictionary<string, float>>();
        Debug.Log("Thresholds loaded successfully.");
    }

    public string GetPredictedEmotion(float[] logits)
    {
        int maxIndex = -1;
        float maxProbability = float.MinValue;

        for (int i = 0; i < logits.Length; i++)
        {
            string emotion = idToLabel[i];
            float probability = logits[i];

            if (probability >= emotionThresholds[emotion])
            {
                Debug.Log($"Emotion: {emotion}. Logit: {logits[i]} Threshold: {emotionThresholds[emotion]}");

                if (probability > maxProbability)
                {
                    maxIndex = i;
                    maxProbability = probability;
                }
            }
        }

        if (maxIndex == -1)
        {
            Debug.Log("No emotion met the threshold.");
            return "No emotion detected";
        }

        return idToLabel[maxIndex];
    }
}