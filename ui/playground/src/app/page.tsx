'use client';

import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Wand2, Video, Box, User, Code, Loader2 } from 'lucide-react';

type Mode = 'segment' | 'track' | 'reconstruct-object' | 'reconstruct-body';

interface SegmentResult {
  instances: Array<{
    instance_id: number;
    concept?: string;
    confidence: number;
    bbox?: number[];
  }>;
  processing_time_ms: number;
}

export default function Playground() {
  const [mode, setMode] = useState<Mode>('segment');
  const [image, setImage] = useState<string | null>(null);
  const [concept, setConcept] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<SegmentResult | null>(null);
  const [showCode, setShowCode] = useState(false);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = () => setImage(reader.result as string);
      reader.readAsDataURL(file);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpg', '.jpeg', '.png', '.webp'] },
    maxFiles: 1,
  });

  const handleSegment = async () => {
    if (!image || !concept) return;
    
    setLoading(true);
    try {
      // In production, this calls the actual API
      // Mock result for demo
      await new Promise(resolve => setTimeout(resolve, 1000));
      setResult({
        instances: [
          { instance_id: 1, concept, confidence: 0.95, bbox: [100, 100, 300, 300] },
          { instance_id: 2, concept, confidence: 0.87, bbox: [400, 150, 550, 350] },
        ],
        processing_time_ms: 45.2,
      });
    } finally {
      setLoading(false);
    }
  };

  const codeSnippet = `from sam3_perception_hub import PerceptionClient

client = PerceptionClient("http://localhost:8080")

result = client.segment(
    image="${image ? 'your_image.jpg' : 'image.jpg'}",
    query=ConceptQuery(
        text="${concept || 'your_concept'}",
        confidence_threshold=0.5
    )
)

for instance in result.instances:
    print(f"Found {instance.concept}: {instance.confidence:.2f}")`;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
              <Box className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">SAM3 Playground</h1>
              <p className="text-xs text-zinc-500">Perception Hub Demo</p>
            </div>
          </div>
          <a 
            href="https://github.com/your-org/sam3-perception-hub" 
            target="_blank"
            className="text-sm text-zinc-400 hover:text-white transition-colors"
          >
            View on GitHub →
          </a>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {/* Mode Tabs */}
        <div className="flex gap-2 mb-8">
          {[
            { id: 'segment', label: 'Segment', icon: Wand2 },
            { id: 'track', label: 'Track Video', icon: Video },
            { id: 'reconstruct-object', label: '3D Object', icon: Box },
            { id: 'reconstruct-body', label: '3D Body', icon: User },
          ].map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setMode(id as Mode)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                mode === id
                  ? 'bg-blue-600 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700 hover:text-zinc-200'
              }`}
            >
              <Icon className="w-4 h-4" />
              {label}
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Input Panel */}
          <div className="space-y-6">
            {/* Image Upload */}
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
                isDragActive
                  ? 'border-blue-500 bg-blue-500/10'
                  : 'border-zinc-700 hover:border-zinc-600 bg-zinc-900/50'
              }`}
            >
              <input {...getInputProps()} />
              {image ? (
                <img
                  src={image}
                  alt="Uploaded"
                  className="max-h-64 mx-auto rounded-lg"
                />
              ) : (
                <div className="space-y-4">
                  <div className="w-16 h-16 mx-auto rounded-full bg-zinc-800 flex items-center justify-center">
                    <Upload className="w-8 h-8 text-zinc-500" />
                  </div>
                  <div>
                    <p className="text-zinc-300 font-medium">Drop an image here</p>
                    <p className="text-sm text-zinc-500">or click to browse</p>
                  </div>
                </div>
              )}
            </div>

            {/* Concept Input */}
            {mode === 'segment' && (
              <div className="space-y-2">
                <label className="text-sm font-medium text-zinc-400">
                  Concept to Segment
                </label>
                <input
                  type="text"
                  value={concept}
                  onChange={(e) => setConcept(e.target.value)}
                  placeholder='e.g., "forklift", "person", "product box"'
                  className="w-full px-4 py-3 bg-zinc-900 border border-zinc-700 rounded-xl text-zinc-100 placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            )}

            {/* Run Button */}
            <button
              onClick={handleSegment}
              disabled={!image || loading}
              className="w-full py-3 px-6 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-xl hover:from-blue-500 hover:to-purple-500 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Processing...
                </>
              ) : (
                <>
                  <Wand2 className="w-5 h-5" />
                  Run Segmentation
                </>
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="space-y-6">
            {/* Results */}
            {result && (
              <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Results</h3>
                  <span className="text-xs text-zinc-500">
                    {result.processing_time_ms.toFixed(1)}ms
                  </span>
                </div>
                <div className="space-y-3">
                  {result.instances.map((inst) => (
                    <div
                      key={inst.instance_id}
                      className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg"
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-blue-600/20 text-blue-400 flex items-center justify-center text-sm font-mono">
                          #{inst.instance_id}
                        </div>
                        <span className="font-medium">{inst.concept}</span>
                      </div>
                      <div className="text-sm text-zinc-400">
                        {(inst.confidence * 100).toFixed(0)}% confidence
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Code Snippet */}
            <div className="bg-zinc-900 border border-zinc-800 rounded-2xl overflow-hidden">
              <button
                onClick={() => setShowCode(!showCode)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-zinc-800/50 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Code className="w-4 h-4 text-zinc-500" />
                  <span className="font-medium">API Code</span>
                </div>
                <span className="text-sm text-zinc-500">
                  {showCode ? 'Hide' : 'Show'}
                </span>
              </button>
              {showCode && (
                <pre className="px-6 pb-6 text-sm overflow-x-auto">
                  <code className="text-green-400">{codeSnippet}</code>
                </pre>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
