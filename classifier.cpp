#include <iostream>
#include <iomanip>
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
    map<string,string> line;

    while (csv >> line)
    {
      string tag = line["tag"];
      string body = line["content"];
      postCount++;
      labelCount[tag]++;

      set<string> bag = split_words(body);
      for (auto &term : bag)
      {
        vocab.insert(term);
        labelWordHits[tag][term]++;
        wordHits[term]++;
      }
    }
  }

  int total_posts() const { return postCount; }
  int vocab_size() const { return vocab.size(); }

  double log_prior(const string &tag) const
  {
    return log(double(labelCount.at(tag)) / postCount);
  }

  double log_likelihood(const string &tag, const string &term) const
  {
    double count = 0;
    if (labelWordHits.at(tag).count(term))
      count = labelWordHits.at(tag).at(term);

    double prob = (count + 1.0) / (labelCount.at(tag) + 2.0);
    return log(prob);
  }

  string guess_label(const string &body, double &bestScore) const
  {
    set<string> bag = split_words(body);
    string chosen;
    bool firstPick = true;

    for (auto &pair : labelCount)
    {
      string tag = pair.first;
      double score = log_prior(tag);

      for (auto &term : bag)
        score += log_likelihood(tag, term);

      if (firstPick || score > bestScore || (fabs(score - bestScore) < 1e-9 && tag < chosen))
      {
        bestScore = score;
        chosen = tag;
        firstPick = false;
      }
    }
    return chosen;
  }

  void show_training_data(const string &file)
  {
    csvstream csv(file);
    map<string,string> row;

    cout << "training data:\n";
    while (csv >> row)
    {
      cout << "  label = " << row["tag"] << ", content = " << row["content"] << "\n";
    }

    cout << "trained on " << postCount << " examples\n";
    cout << "vocabulary size = " << vocab.size() << "\n\n";
  }

  void show_classes() const
  {
    cout << "classes:\n";
    for (auto &pair : labelCount)
    {
      cout << "  " << pair.first << ", " << pair.second << " examples, log-prior = "
           << fixed << setprecision(3) << log_prior(pair.first) << "\n";
    }
    cout << "\n";
  }

  void show_params() const
  {
    cout << "classifier parameters:\n";
    for (auto &outer : labelWordHits)
    {
      for (auto &inner : outer.second)
      {
        cout << "  " << outer.first << ":" << inner.first << ", count = " << inner.second
             << ", log-likelihood = " << fixed << setprecision(3)
             << log_likelihood(outer.first, inner.first) << "\n";
      }
    }
    cout << "\n";
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
  catch (const csvstream_exception &e)
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
    map<string,string> post;

    cout << "test data:\n";
    int right = 0, total = 0;

    while (csv >> post)
    {
      string correctTag = post["tag"];
      string text = post["content"];
      double score;
      string predicted = model.guess_label(text, score);

      cout << "  correct = " << correctTag << ", predicted = " << predicted
           << ", log-probability score = " << score << "\n";
      cout << "  content = " << text << "\n\n";

      if (predicted == correctTag) right++;
      total++;
    }

    cout << "performance: " << right << " / " << total << " posts predicted correctly\n";
  }
  catch (const csvstream_exception &e)
  {
    cout << "Error opening file: " << testFile << endl;
    return 1;
  }
}
