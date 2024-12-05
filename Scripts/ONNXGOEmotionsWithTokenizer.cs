using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Unity;

public class ONNXGoEmotionsWithTokenizer : MonoBehaviour
{
    public string inputText = "I feel so many emotions";
    public string modelPath = "Assets/Resources/EmotionDetector/roberta-base-go_emotions-onnx/onnx/model.onnx";
    private InferenceSession session;

    private void Update()
    {
        if (Input.GetKeyDown(KeyCode.Keypad0))
        {
            // Initialize ONNX Runtime session
            session = new InferenceSession(modelPath);

            // Prepare input tensor (raw text for embedded tokenizer)
            var inputTensor = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_text", PrepareTextTensor(inputText))
            };

            // Run inference
            var results = session.Run(inputTensor);

            // Process output logits
            var logits = results.First().AsTensor<float>().ToArray();
            var classifiedLabels = ApplyThresholds(logits);

            foreach (var label in classifiedLabels)
            {
                Debug.Log($"Label: {label.Key}, Probability: {label.Value}");
            }
        }
    }

    private Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string> PrepareTextTensor(string text)
    {
        // Create a tensor for the input text
        return new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<string>(new[] { text }, new[] { 1 });
    }

    private Dictionary<string, float> ApplyThresholds(float[] logits)
    {
        // Define thresholds for each label
        Dictionary<string, float> labelThresholds = new Dictionary<string, float>
        {
            { "admiration", 0.25f }, { "amusement", 0.45f }, { "anger", 0.15f },
            { "annoyance", 0.10f }, { "approval", 0.30f }, { "caring", 0.40f },
            { "confusion", 0.55f }, { "curiosity", 0.25f }, { "desire", 0.25f },
            { "disappointment", 0.40f }, { "disapproval", 0.30f }, { "disgust", 0.20f },
            { "embarrassment", 0.10f }, { "excitement", 0.35f }, { "fear", 0.40f },
            { "gratitude", 0.45f }, { "grief", 0.05f }, { "joy", 0.40f },
            { "love", 0.25f }, { "nervousness", 0.25f }, { "optimism", 0.20f },
            { "pride", 0.10f }, { "realization", 0.15f }, { "relief", 0.05f },
            { "remorse", 0.10f }, { "sadness", 0.40f }, { "surprise", 0.15f },
            { "neutral", 0.25f }
        };

        List<string> labelNames = labelThresholds.Keys.ToList();
        Dictionary<string, float> results = new Dictionary<string, float>();

        // Apply thresholds
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > labelThresholds[labelNames[i]])
            {
                results[labelNames[i]] = logits[i];
            }
        }

        return results;
    }

    void OnDestroy()
    {
        // Dispose the session to free resources
        session.Dispose();
    }
}