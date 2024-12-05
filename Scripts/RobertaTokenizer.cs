using System;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json.Linq;
using UnityEngine;

public class RobertaTokenizer : MonoBehaviour
{
    [Header("Drag Tokenizer Files Here")]
    public TextAsset tokenizerJsonFile;

    private Dictionary<string, int> vocab;
    private int bosTokenId;
    private int padTokenId;
    private int eosTokenId;
    private int unkTokenId;
    private int maxLength;

    void Start()
    {
        if (tokenizerJsonFile == null)
        {
            Debug.LogError("Tokenizer file not assigned!");
            return;
        }

        LoadTokenizer();
    }

    private void LoadTokenizer()
    {
        try
        {
            var tokenizerJson = JObject.Parse(tokenizerJsonFile.text);

            // Load vocabulary
            vocab = tokenizerJson["model"]["vocab"].ToObject<Dictionary<string, int>>();

            // Load special tokens
            bosTokenId = vocab["<s>"];
            padTokenId = vocab["<pad>"];
            eosTokenId = vocab["</s>"];
            unkTokenId = vocab["<unk>"];

            // Load truncation settings
            maxLength = (int)tokenizerJson["truncation"]["max_length"];

            Debug.Log("Tokenizer loaded successfully.");
        }
        catch (Exception ex)
        {
            Debug.LogError($"Error loading tokenizer: {ex.Message}");
        }
    }

    public List<int> Tokenize(string text)
    {
        if (vocab == null)
        {
            Debug.LogError("Tokenizer not loaded!");
            return new List<int>();
        }

        // Tokenize text into tokens
        var tokens = text.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        var tokenIds = new List<int>();

        foreach (var token in tokens)
        {
            if (vocab.TryGetValue(token, out int tokenId))
                tokenIds.Add(tokenId);
            else
                tokenIds.Add(unkTokenId);
        }

        // Add special tokens
        tokenIds.Insert(0, bosTokenId); // Add <s>
        tokenIds.Add(eosTokenId); // Add </s>

        // Truncate to max length
        if (tokenIds.Count > maxLength)
        {
            tokenIds = tokenIds.Take(maxLength).ToList();
            tokenIds[tokenIds.Count - 1] = eosTokenId; // Ensure end token remains
        }

        return tokenIds;
    }

    public Dictionary<string, object> CreateInput(string text)
    {
        var tokenIds = Tokenize(text);

        // Pad to max length
        while (tokenIds.Count < maxLength)
            tokenIds.Add(padTokenId);

        // Create attention mask
        var attentionMask = tokenIds.Select(id => id != padTokenId ? 1 : 0).ToList();

        return new Dictionary<string, object>
        {
            { "input_ids", tokenIds },
            { "attention_mask", attentionMask }
        };
    }
}