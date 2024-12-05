// File: EmotionPipeline.cs
using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Linq;

[RequireComponent(typeof(EmotionTokenizer), typeof(EmotionRecognition))]
public class EmotionPipeline : MonoBehaviour
{
    public EmotionTokenizer tokenizer;
    public ModelAsset onnxModel;
    public EmotionRecognition recognitionModel;

    [SerializeField]
    KeyCode testKey;

    [SerializeField]
    private string testInput = "I feel fantastic today!";

    private IWorker worker;

    private void Update()
    {
        if (Input.GetKeyDown(testKey))
        {
            StartTest();
        }
    }

    private void Start()
    {
        if (tokenizer == null || recognitionModel == null || onnxModel == null)
        {
            Debug.LogError("Tokenizer, Recognition Model, or ONNX model not assigned!");
            return;
        }

        Model runtimeModel = ModelLoader.Load(onnxModel);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }

    async void StartTest()
    {
        int[] inputIds = await tokenizer.Tokenize(testInput);
        if (inputIds == null)
        {
            Debug.LogError("Tokenization failed!");
            return;
        }

        float[] attentionMask = inputIds.Select(id => id != 0 ? 1.0f : 0.0f).ToArray();

        float[] logits = await Predict(inputIds, attentionMask);

        float[] probabilities = logits.Select(Softmaxize).ToArray();
        string emotion = recognitionModel.GetPredictedEmotion(probabilities);

        Debug.Log($"Input Text: {testInput}");
        Debug.Log($"Predicted Emotion: {emotion}");
    }

    private async Task<float[]> Predict(int[] inputIds, float[] attentionMask)
    {
        TensorInt inputTensor = new TensorInt(new TensorShape(1, inputIds.Length), inputIds);
        TensorFloat maskTensor = new TensorFloat(new TensorShape(1, attentionMask.Length), attentionMask);

        worker.Execute(new Dictionary<string, Tensor>
        {
            { "input_ids", inputTensor },
            { "attention_mask", maskTensor }
        });

        TensorFloat outputTensor = (TensorFloat)worker.PeekOutput();
        await outputTensor.CompleteOperationsAndDownloadAsync();
        float[] logits = outputTensor.ToReadOnlyArray();

        inputTensor.Dispose();
        maskTensor.Dispose();
        outputTensor.Dispose();

        return logits;
    }

    float Softmaxize(float x)
    {
        float expX = Mathf.Exp(x);
        return expX / (1.0f + expX);
    }

    void OnDestroy()
    {
        if (worker != null)
        {
            worker.Dispose();
        }
    }
}
