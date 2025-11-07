#include "csvstream.hpp"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
using namespace std;

static string show3(double v)
{
  ostringstream oss;
  oss.setf(ios::fixed);
  oss << setprecision(3) << v;
  return oss.str();
}

class NBClassifier
{
public:
  map<string,int> labelCount;
  map<string,map<string,int>> labelWordHits;
  set<string> vocab;
  map<string,int> wordHits;

  void train(const string &file)
  {
    try
    {
      csvstream csv(file);
      map<string,string> row;

      cout << "training data:\n";

      while (csv >> row)
      {
        string label = row["tag"];
        string text = row["content"];
        cout << "  label = " << label
             << ", content = " << text << "\n";

        labelCount[label]++;

        set<string> words = split_words(text);
        set<string> seen;

        for (const string &word : words)
        {
          if (!seen.count(word))
          {
            labelWordHits[label][word]++;
            vocab.insert(word);
            wordHits[word]++;
            seen.insert(word);
          }
        }
      }

      cout << "trained on "
           << total_examples() << " examples\n";

      cout << "vocabulary size = "
           << vocab.size() << "\n\n";
    }
    catch (const csvstream_exception &)
    {
      cout << "Error opening file: " << file << endl;
      throw;
    }
  }

  void print_classes() const
  {
    cout << "classes:\n";

    for (auto &p : labelCount)
    {
      cout << "  " << p.first
           << ", " << p.second
           << " examples, log-prior = "
           << show3(log_prior(p.first)) << "\n";
    }

    cout << "\n";
  }

  void print_params() const
  {
    cout << "classifier parameters:\n";

    for (auto &lp : labelWordHits)
    {
      const string &label = lp.first;

      for (auto &wp : lp.second)
      {
        cout << "  " << label
             << ":" << wp.first
             << ", count = " << wp.second
             << ", log-likelihood = "
             << show3(log_prob_word(label, wp.first))
             << "\n";
      }
    }

    cout << "\n";
  }

  string predict(const string &text,
                 double &bestScore) const
  {
    set<string> bag = split_words(text);
    string best;
    bool first = true;

    for (auto &lbl : labelCount)
    {
      const string &label = lbl.first;
      double score = log_prior(label);

      for (const string &w : vocab)
      {
        double p = prob_word_given_label(label, w);

        if (bag.count(w))
          score += log(p);
        else
          score += log(1.0 - p);
      }

      if (first || score > bestScore ||
          (fabs(score - bestScore) < 1e-9 &&
           label < best))
      {
        bestScore = score;
        best = label;
        first = false;
      }
    }

    return best;
  }

private:
  static set<string> split_words(const string &line)
  {
    set<string> words;
    string w;

    for (char c : line)
    {
      if (isalpha(c) || isdigit(c))
        w += tolower(c);
      else if (!w.empty())
      {
        words.insert(w);
        w.clear();
      }
    }

    if (!w.empty()) words.insert(w);
    return words;
  }

  int total_examples() const
  {
    int n = 0;
    for (auto &p : labelCount) n += p.second;
    return n;
  }

  double log_prior(const string &label) const
  {
    double n = labelCount.at(label);
    double t = total_examples();
    return log(n / t);
  }

  double log_prob_word(const string &label,
                       const string &word) const
  {
    int hits = 0;
    auto itL = labelWordHits.find(label);
    if (itL != labelWordHits.end())
    {
      auto itW = itL->second.find(word);
      if (itW != itL->second.end())
        hits = itW->second;
    }

    double denom = double(labelCount.at(label));
    double prob = double(hits) / denom;
    return log(prob);
  }

  double prob_word_given_label(const string &label,
                               const string &word) const
  {
    double labelDocs = double(labelCount.at(label));
    double hit = 0.0;

    auto itL = labelWordHits.find(label);
    if (itL != labelWordHits.end())
    {
      auto itW = itL->second.find(word);
      if (itW != itL->second.end())
        hit = double(itW->second);
    }

    return (hit + 1.0) / (labelDocs + 2.0);
  }
};

static void run_tests(NBClassifier &clf,
                      const string &file)
{
  try
  {
    csvstream csv(file);
    map<string,string> row;
    cout << "test data:\n";
    int correct = 0, total = 0;

    while (csv >> row)
    {
      string lbl = row["tag"];
      string txt = row["content"];
      double score = 0.0;
      string pred = clf.predict(txt, score);

      cout << "  correct = " << lbl
           << ", predicted = " << pred
           << ", log-probability score = "
           << round(score * 10.0) / 10.0
           << "\n";

      cout << "  content = "
           << txt << "\n\n";

      if (pred == lbl) correct++;
      total++;
    }

    cout << "performance: "
         << correct << " / " << total
         << " posts predicted correctly\n";
  }
  catch (const csvstream_exception &)
  {
    cout << "Error opening file: " << file << endl;
  }
}

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    cout << "Usage: classifier.exe TRAIN_FILE"
         << " [TEST_FILE]" << endl;
    return 1;
  }

  string train_file = argv[1];
  NBClassifier clf;

  try
  {
    clf.train(train_file);
  }
  catch (const csvstream_exception &)
  {
    return 1;
  }

  if (argc == 2)
  {
    clf.print_classes();
    clf.print_params();
    return 0;
  }

  string test_file = argv[2];
  run_tests(clf, test_file);
  return 0;
}
