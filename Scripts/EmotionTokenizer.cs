using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Newtonsoft.Json;
using Tallow;
using UnityEngine;
using System.Threading.Tasks;

public class EmotionTokenizer : MonoBehaviour
{
    [Header("Tokenizer Files")]
    public TextAsset vocabFile;
    public TextAsset mergesFile;

    private Dictionary<string, int> vocab;
    private Dictionary<int, string> reverseVocab;
    private Dictionary<string, int> bpeRanks;
    private Dictionary<string, string> cache;

    private int maxLength = 128;

    private int bosTokenId;
    private int eosTokenId;
    private int padTokenId;
    private int unkTokenId;

    void Start()
    {
        if (vocabFile == null || mergesFile == null)
        {
            Debug.LogError("Tokenizer files are not assigned!");
            return;
        }

        InitializeTokenizer();
    }

    private void InitializeTokenizer()
    {
        vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabFile.text);
        reverseVocab = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);

        var merges = mergesFile.text.Split(new[] { "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries)
            .Skip(1)
            .Select(line => line.Split(' '))
            .ToArray();

        bpeRanks = merges.Select((pair, idx) => new KeyValuePair<string, int>(string.Join(" ", pair), idx))
            .ToDictionary(kv => kv.Key, kv => kv.Value);

        bosTokenId = vocab["<s>"];
        eosTokenId = vocab["</s>"];
        padTokenId = vocab["<pad>"];
        unkTokenId = vocab["<unk>"];

        cache = new Dictionary<string, string>();
        Debug.Log("Tokenizer initialized.");
    }

    public async Task<int[]> Tokenize(string oldText)
    {
        // Prepare tokens list from input string
        var tokens = await oldText.GetWordsFromString();

        // Apply BPE to each token and add Ġ only for tokens after the first one
        var bpeTokens = tokens.SelectMany((token, index) =>
        {
            // Apply BPE
            var subwords = Bpe(token).ToArray();

            if (index > 0)
            {
                subwords[0] = "Ġ" + subwords[0];
            }

            return subwords;
        })

        // Convert tokens to IDs using vocab mapping
        .Select(token => vocab.ContainsKey(token) ? vocab[token] : unkTokenId).ToList();

        // Add special tokens
        bpeTokens.Insert(0, bosTokenId); // Add <s> at the start
        bpeTokens.Add(eosTokenId); // Add </s> at the end

        // Apply truncation
        return ApplyTruncation(bpeTokens);
    }

    public string Decode(int[] tokenIds)
    {
        // Join tokens into a single string
        var decodedString = string.Join("", tokenIds
            .Where(id => id != padTokenId && id != bosTokenId && id != eosTokenId)
            .Select(id => reverseVocab.ContainsKey(id) ? reverseVocab[id] : "<unk>"));

        // Replace "Ġ" with a space
        return decodedString.Replace("Ġ", " ");
    }

    private IEnumerable<string> Bpe(string token)
    {
        if (cache.ContainsKey(token))
        {
            return cache[token].Split(' ');
        }

        var word = token.Select(c => c.ToString()).ToArray();
        var pairs = GetPairs(word);

        while (pairs.Count > 0)
        {
            var bigram = pairs.OrderBy(p => bpeRanks.ContainsKey(string.Join(" ", p)) ? bpeRanks[string.Join(" ", p)] : int.MaxValue).FirstOrDefault();
            if (!bpeRanks.ContainsKey(string.Join(" ", bigram)))
                break;

            var first = bigram[0];
            var second = bigram[1];
            var newWord = new List<string>();

            int i = 0;
            while (i < word.Length)
            {
                int j = Array.IndexOf(word, first, i);
                if (j == -1)
                {
                    newWord.AddRange(word.Skip(i));
                    break;
                }

                newWord.AddRange(word.Skip(i).Take(j - i));
                if (j < word.Length - 1 && word[j] == first && word[j + 1] == second)
                {
                    newWord.Add(first + second);
                    i = j + 2;
                }
                else
                {
                    newWord.Add(word[j]);
                    i = j + 1;
                }
            }

            word = newWord.ToArray();
            pairs = GetPairs(word);
        }

        var result = string.Join(" ", word);
        cache[token] = result;
        return word;
    }

    private List<string[]> GetPairs(string[] word)
    {
        var pairs = new List<string[]>();
        for (int i = 0; i < word.Length - 1; i++)
        {
            pairs.Add(new[] { word[i], word[i + 1] });
        }
        return pairs;
    }

    private int[] ApplyTruncation(List<int> tokenIds)
    {
        if (tokenIds.Count > maxLength)
        {
            tokenIds = tokenIds.Take(maxLength).ToList();
            tokenIds[tokenIds.Count - 1] = eosTokenId;
        }

        return tokenIds.ToArray();
    }
}