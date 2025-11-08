#include "csvstream.hpp"
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <cmath>
#include <sstream>
#include <iomanip>
using namespace std;

class NBClassifier
{
private:
  map<string, int> labelCount;
  map<string, map<string, int>> labelWordHits;
  set<string> vocab;
  map<string, int> wordHits;

public:
  void train(const string &file)
  {
    try
    {
      csvstream csv(file);
      map<string, string> row;
      cout << "training data:\n";
      while (csv >> row)
      {
        string label = row["tag"];
        string text = row["content"];
        cout << "  label = " << label << ", content = " << text << "\n";
        labelCount[label]++;
        istringstream source(text);
        set<string> words;
        string word;
        while (source >> word)
          words.insert(word);
        for (const string &w : words)
        {
          labelWordHits[label][w]++;
          vocab.insert(w);
          wordHits[w]++;
        }
      }
      int total = 0;
      for (auto &p : labelCount)
        total += p.second;
      cout << "trained on " << total << " examples\n";
      cout << "vocabulary size = " << vocab.size() << "\n\n";
    }
    catch (const csvstream_exception &e)
    {
      cout << "Error opening file: " << file << endl;
    }
  }

  void print_classes() const
  {
    cout << "classes:\n";
    int total = 0;
    for (auto &p : labelCount)
      total += p.second;
    for (auto &p : labelCount)
    {
      double prior = log(double(p.second) / double(total));
      ostringstream out;
      out << fixed << setprecision(3) << prior;
      cout << "  " << p.first << ", " << p.second << " examples, log-prior = " << out.str() << "\n";
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
        const string &word = wp.first;
        int hits = wp.second;
        double n_label = double(labelCount.at(label));
        double prob;
        if (hits > 0)
          prob = hits / n_label;
        else
          prob = 1.0 / (n_label + 2.0);
        double loglike = log(prob);
        ostringstream out;
        out << fixed << setprecision(3) << loglike;
        cout << "  " << label << ":" << word << ", count = " << hits << ", log-likelihood = " << out.str() << "\n";
      }
    }
    cout << "\n";
  }

  string predict(const string &text, double &bestScore) const
  {
    set<string> bag;
    istringstream in(text);
    string w;
    while (in >> w)
      bag.insert(w);
    string best;
    bool first = true;
    for (auto &lbl : labelCount)
    {
      const string &label = lbl.first;
      double labelDocs = double(labelCount.at(label));
      int total = 0;
      for (auto &p : labelCount)
        total += p.second;
      double score = log(labelDocs / double(total));
      for (const string &w : vocab)
      {
        int hits = 0;
        auto itL = labelWordHits.find(label);
        if (itL != labelWordHits.end())
        {
          auto itW = itL->second.find(w);
          if (itW != itL->second.end())
            hits = itW->second;
        }
        double p;
        if (hits > 0)
          p = hits / labelDocs;
        else
          p = 1.0 / (labelDocs + 2.0);
        if (bag.count(w))
          score += log(p);
        else
          score += log(1.0 - p);
      }
      if (first || score > bestScore || (fabs(score - bestScore) < 1e-9 && label < best))
      {
        bestScore = score;
        best = label;
        first = false;
      }
    }
    return best;
  }
};

void run_tests(NBClassifier &clf, const string &file)
{
  try
  {
    csvstream csv(file);
    map<string, string> row;
    cout << "test data:\n";
    int correct = 0, total = 0;
    while (csv >> row)
    {
      string lbl = row["tag"];
      string txt = row["content"];
      double score = 0.0;
      string pred = clf.predict(txt, score);
      ostringstream out;
      out << fixed << setprecision(3) << score;
      cout << "  correct = " << lbl << ", predicted = " << pred << ", log-probability score = " << out.str() << "\n";
      cout << "  content = " << txt << "\n\n";
      if (pred == lbl)
        correct++;
      total++;
    }
    cout << "performance: " << correct << " / " << total << " posts predicted correctly\n\n";
  }
  catch (const csvstream_exception &e)
  {
    cout << "Error opening file: " << file << endl;
  }
}

int main(int argc, char *argv[])
{
  cout << fixed << setprecision(3);
  if (argc != 2 && argc != 3)
  {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]\n";
    return 1;
  }
  NBClassifier clf;
  string train_file = argv[1];
  clf.train(train_file);
  if (argc == 2)
  {
    clf.print_classes();
    clf.print_params();
  }
  else
  {
    string test_file = argv[2];
    run_tests(clf, test_file);
  }
  return 0;
}
