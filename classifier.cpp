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
  void train(const string &file, bool verbose)
  {
    csvstream csv(file);
    map<string, string> row;
    if (verbose)
      cout << "training data:\n";
    while (csv >> row)
    {
      string label = row["tag"];
      string text = row["content"];
      if (verbose)
      {
        cout << "  label = " << label << ", content = " << text << "\n";
      }
      labelCount[label]++;
      istringstream in(text);
      set<string> words;
      string w;
      while (in >> w)
        words.insert(w);
      for (const string &x : words)
      {
        labelWordHits[label][x]++;
        vocab.insert(x);
        wordHits[x]++;
      }
    }
    int total = 0;
    for (auto &p : labelCount)
      total += p.second;
    cout << "trained on " << total << " examples\n";
    if (verbose)
    {
      cout << "vocabulary size = " << vocab.size() << "\n\n";
    }
    else
    {
      cout << "\n";
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
      cout << "  " << p.first << ", " << p.second
           << " examples, log-prior = " << prior << "\n";
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
        double prob = (hits > 0) ? (hits / n_label) : (1.0 / (n_label + 2.0));
        double loglike = log(prob);
        cout << "  " << label << ":" << word << ", count = " << hits
             << ", log-likelihood = " << loglike << "\n";
      }
    }
    cout << "\n";
  }

  string predict(const string &text, double &bestScore) const
  {
    set<string> bag;
    {
      istringstream in(text);
      string w;
      while (in >> w)
        bag.insert(w);
    }
    string best;
    bool first = true;

    int total = 0;
    for (auto &p : labelCount)
      total += p.second;

    for (auto &lbl : labelCount)
    {
      const string &label = lbl.first;
      double n_label = double(labelCount.at(label));
      double score = log(n_label / double(total));
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
        double p = (hits > 0) ? (hits / n_label) : (1.0 / (n_label + 2.0));
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

static void run_tests(NBClassifier &clf, const string &file)
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
    cout.setf(ios::fixed);
    cout << setprecision(1);
    cout << "  correct = " << lbl << ", predicted = " << pred
         << ", log-probability score = " << score << "\n";
    cout << "  content = " << txt << "\n\n";
    if (pred == lbl)
      correct++;
    total++;
  }
  cout << "performance: " << correct << " / " << total
       << " posts predicted correctly\n";
}

int main(int argc, char *argv[])
{
  if (argc != 2 && argc != 3)
  {
    cout << "Usage: classifier.exe TRAIN_FILE [TEST_FILE]\n";
    return 1;
  }
  try
  {
    NBClassifier clf;
    string train_file = argv[1];
    if (argc == 2)
    {
      clf.train(train_file, true);
      clf.print_classes();
      clf.print_params();
    }
    else
    {
      clf.train(train_file, false);
      string test_file = argv[2];
      run_tests(clf, test_file);
    }
  }
  catch (const csvstream_exception &e)
  {
    size_t k = e.msg.find(": ");
    string tail = (k == string::npos) ? e.msg : e.msg.substr(k + 2);
    cout << "Error opening file: " << tail << "\n";
    return 1;
  }
  return 0;
}
