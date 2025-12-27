'use client';

import { useState } from 'react';

export default function Home() {
  const [text, setText] = useState('');
  const [result, setResult] = useState('');
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
    } catch (error) {
      setResult('Error: Could not connect to backend');
    }
    setLoading(false);
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-zinc-50 font-sans dark:bg-black">
      <main className="flex min-h-screen w-full max-w-3xl flex-col items-center justify-between py-32 px-16 bg-white dark:bg-black sm:items-start">
        <div className="flex flex-col items-center gap-6 text-center sm:items-start sm:text-left">
          <h1 className="max-w-xs text-3xl font-semibold leading-10 tracking-tight text-black dark:text-zinc-50">
            Vietnamese Fake News Detector
          </h1>
          <p className="max-w-md text-lg leading-8 text-zinc-600 dark:text-zinc-400">
            Enter a news article in Vietnamese to check if it's real or fake.
          </p>
          <form onSubmit={handleSubmit} className="w-full max-w-md">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your news article here..."
              className="w-full h-32 p-4 border border-zinc-300 rounded-md resize-none"
              required
            />
            <button
              type="submit"
              disabled={loading}
              className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? 'Analyzing...' : 'Check News'}
            </button>
          </form>
          {result && (
            <div className="mt-6 p-4 bg-zinc-100 dark:bg-zinc-800 rounded-md">
              <p className="text-lg font-medium">{result}</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
