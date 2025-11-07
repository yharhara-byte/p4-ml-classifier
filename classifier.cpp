#include <iostream>
#include <string>
#include <map>
#include <set>
#include <cmath>
#include <sstream>
#include "csvstream.hpp"
using namespace std;

set<string> split_words(const string &text)
{
  istringstream in(text);
  set<string> bag;
  string piece;
  while (in >> piece) bag.insert(piece);
  return bag;
}

class Classifier
{
private:
  int postCount = 0;
  set<string> vocab;
  map<string,int> labelCount;
  map<string,map<string,int>> labelWordHits;
  map<string,int> wordHits;

public:
  void train(const string &file)
  {
    csvstream csv(file);
    map<string,string> row;

    while (csv >> row)
    {
      string tag = row.at("tag");
      string body = row.at("content");

      postCount++;
      labelCount[tag]++;

      set<string> bag = split_words(body);
      for (const auto &term : bag)
      {
        vocab.insert(term);
        labelWordHits[tag][term]++;
        wordHits[term]++;
      }
    }
  }

  void show_training_data(const string &file) const
  {
    csvstream csv(file);
    map<string,string> row;

    cout << "training data:\n";
    while (csv >> row)
    {
      cout << "  label = " << row.at("tag")
           << ", content = " << row.at("content") << "\n";
    }

    cout << "trained on " << postCount << " examples\n";
    cout << "vocabulary size = " << vocab.size() << "\n\n";
  }

  void show_classes() const
  {
    cout << "classes:\n";
    for (const auto &p : labelCount)
    {
      const string &tag = p.first;
      double prior = log(double(p.second) / double(postCount));
      cout << "  " << tag << ", " << p.second
           << " examples, log-prior = " << prior << "\n";
    }
    cout << "\n";
  }

  void show_params() const
  {
    cout << "classifier parameters:\n";
    for (const auto &outer : labelWordHits)
    {
      const string &tag = outer.first;
      for (const auto &inner : outer.second)
      {
        const string &term = inner.first;
        int hit = inner.second;
        double ll = log(double(hit) / double(labelCount.at(tag)));
        cout << "  " << tag << ":" << term
             << ", count = " << hit
             << ", log-likelihood = " << ll << "\n";
      }
    }
    cout << "\n";
  }

  double log_prior(const string &tag) const
  {
    return log(double(labelCount.at(tag)) / double(postCount));
  }

  double word_ll_for_predict(const string &tag, const string &term) const 
  {
    int wordHit = 0;

    auto itLabel = labelWordHits.find(tag);
    if (itLabel != labelWordHits.end()) 
    {
      auto itTerm = itLabel->second.find(term);
      if (itTerm != itLabel->second.end()) 
      {
        wordHit = itTerm->second;
      }
    }

    int totalHits = 0;
    for (const auto &p : labelWordHits.at(tag)) 
    {
      totalHits += p.second;
    }

    double prob = (wordHit + 1.0) / (totalHits + double(vocab.size()));

    return log10(prob); 
  }

  string guess_label(const string &body, double &bestScore) const
  {
    set<string> bag = split_words(body);
    string choice;
    bool firstPick = true;

    for (const auto &pair : labelCount)
    {
      const string &tag = pair.first;
      double score = log_prior(tag);

      for (const auto &term : bag)
      {
        score += word_ll_for_predict(tag, term);
      }

      if (firstPick ||
          score > bestScore ||
          (fabs(score - bestScore) < 1e-9 && tag < choice))
      {
        bestScore = score;
        choice = tag;
        firstPick = false;
      }
    }

    return choice;
  }
};

int main(int argc, char* argv[])
{
  cout.precision(3);

  if (argc != 2 && argc != 3)
  {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]" << endl;
    return 1;
  }

  string trainFile = argv[1];
  string testFile;
  bool hasTest = argc == 3;
  if (hasTest) testFile = argv[2];

  Classifier model;

  try
  {
    model.train(trainFile);
  }
  catch (const csvstream_exception &)
  {
    cout << "Error opening file: " << trainFile << endl;
    return 1;
  }

  model.show_training_data(trainFile);

  if (!hasTest)
  {
    model.show_classes();
    model.show_params();
    return 0;
  }

  try
  {
    csvstream csv(testFile);
    map<string,string> row;

    cout << "test data:\n";
    int right = 0;
    int total = 0;

    while (csv >> row)
    {
      string correctTag = row.at("tag");
      string text = row.at("content");
      double score = 0.0;
      string predicted = model.guess_label(text, score);

      cout << "  correct = " << correctTag
           << ", predicted = " << predicted
           << ", log-probability score = " << score << "\n";

      cout << "  content = " << text << "\n\n";

      if (predicted == correctTag) right++;
      total++;
    }

    cout << "performance: " << right
         << " / " << total
         << " posts predicted correctly\n";
  }
  catch (const csvstream_exception &)
  {
    cout << "Error opening file: " << testFile << endl;
    return 1;
  }
}
