'use client';

import { useState } from 'react';

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState('');
  const [confidence, setConfidence] = useState<number[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });
      const data = await response.json();
      setResult(data.prediction);
      setConfidence(data.confidence);
    } catch {
      setResult('Error: Could not connect to backend');
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-linear-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-16">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-4">
            Vietnamese Fake News Detector
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-2xl mx-auto">
            Enter a news article in Vietnamese to check if it is real or fake. Our AI-powered tool analyzes the content for authenticity.
          </p>
        </header>

        <main className="max-w-4xl mx-auto">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-xl p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label htmlFor="news-text" className="block text-lg font-medium text-gray-700 dark:text-gray-300 mb-2">
                  News Article Text
                </label>
                <textarea
                  id="news-text"
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  placeholder="Paste your news article here..."
                  className="w-full h-40 p-4 border border-gray-300 dark:border-gray-600 rounded-lg resize-none focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white transition duration-200"
                  required
                />
              </div>

              <button
                type="submit"
                disabled={loading}
                className="w-full bg-linear-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <svg className="animate-spin h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    <span>Analyzing...</span>
                  </>
                ) : (
                  <span>Check News</span>
                )}
              </button>
            </form>

            {result && (
              <div className="mt-8 p-6 bg-gray-50 dark:bg-gray-700 rounded-lg border-l-4 border-blue-500">
                <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Analysis Result</h3>
                <p className="text-lg text-gray-700 dark:text-gray-300 mb-4">
                  Prediction: <span className={`font-bold ${result === 'Real News' ? 'text-green-600' : 'text-red-600'}`}>{result}</span>
                </p>
                {confidence.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm text-gray-600 dark:text-gray-400">Confidence Levels:</p>
                    <div className="flex space-x-4">
                      <div className="flex-1">
                        <div className="flex justify-between text-sm mb-1">
                          <span>Real News</span>
                          <span>{typeof confidence[0] === 'number' ? confidence[0].toFixed(1) : 'N/A'}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-green-500 h-2 rounded-full" style={{ width: `${typeof confidence[0] === 'number' ? confidence[0] : 0}%` }}></div>
                        </div>
                      </div>
                      <div className="flex-1">
                        <div className="flex justify-between text-sm mb-1">
                          <span>Fake News</span>
                          <span>{typeof confidence[1] === 'number' ? confidence[1].toFixed(1) : 'N/A'}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div className="bg-red-500 h-2 rounded-full" style={{ width: `${typeof confidence[1] === 'number' ? confidence[1] : 0}%` }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </main>

        <footer className="text-center mt-16 text-gray-500 dark:text-gray-400">
          <p>&copy; 2025 Vietnamese Fake News Detector. Built with Next.js and AI.</p>
        </footer>
      </div>
    </div>
  );
}
