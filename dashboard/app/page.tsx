"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import * as d3 from "d3";

interface GPU {
  index: number;
  name: string;
  memory_total_mb: number;
  memory_used_mb: number;
  memory_free_mb: number;
  utilization_percent: number;
  task_id: string | null;
}

interface Task {
  task_id: string;
  name?: string;
  status: string;
  task_type?: string;
  gpu_id: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  stdout: string | null;
  stderr: string | null;
  return_code: number | null;
  error: string | null;
  progress?: number;
  progress_message?: string;
  model_id?: string;
}

interface Model {
  model_id: string;
  name: string;
  task_id: string | null;
  model_type: string | null;
  framework: string;
  created_at: string;
  updated_at: string | null;
  model_path: string | null;
  description: string | null;
  status: string;
  file_size: number | null;
}

interface MetricHistory {
  timestamp: number;
  values: number[];
}

// Circular Gauge Component
function GaugeChart({ value, max, label, color, size = 120 }: {
  value: number; max: number; label: string; color: string; size?: number
}) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = size;
    const height = size;
    const radius = Math.min(width, height) / 2 - 10;
    const thickness = 12;

    const g = svg
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2}, ${height / 2})`);

    // Background arc
    const bgArc = d3.arc<unknown>()
      .innerRadius(radius - thickness)
      .outerRadius(radius)
      .startAngle(0)
      .endAngle(2 * Math.PI);

    g.append("path")
      .attr("d", bgArc({}) || "")
      .attr("fill", "#1f2937");

    // Value arc
    const percentage = Math.min(value / max, 1);
    const valueArc = d3.arc<unknown>()
      .innerRadius(radius - thickness)
      .outerRadius(radius)
      .startAngle(0)
      .cornerRadius(6);

    g.append("path")
      .datum({ endAngle: 0 })
      .attr("d", (d: { endAngle: number }) => valueArc({ ...d, startAngle: 0 }) || "")
      .attr("fill", `url(#gradient-${label.replace(/\s/g, "")})`)
      .transition()
      .duration(750)
      .attrTween("d", function() {
        const interpolate = d3.interpolate(0, percentage * 2 * Math.PI);
        return function(t: number) {
          return valueArc({ startAngle: 0, endAngle: interpolate(t) }) || "";
        };
      });

    // Gradient definition
    const defs = svg.append("defs");
    const gradient = defs.append("linearGradient")
      .attr("id", `gradient-${label.replace(/\s/g, "")}`)
      .attr("x1", "0%").attr("y1", "0%")
      .attr("x2", "100%").attr("y2", "100%");
    gradient.append("stop").attr("offset", "0%").attr("stop-color", color);
    gradient.append("stop").attr("offset", "100%").attr("stop-color", d3.color(color)?.darker(0.5)?.toString() || color);

    // Center text
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "0.1em")
      .attr("fill", "#fff")
      .attr("font-size", size / 4)
      .attr("font-weight", "bold")
      .text(`${Math.round(percentage * 100)}%`);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.8em")
      .attr("fill", "#9ca3af")
      .attr("font-size", size / 10)
      .text(label);

  }, [value, max, label, color, size]);

  return <svg ref={svgRef} />;
}

// Line Chart Component for History
function LineChart({ data, colors, height = 150 }: {
  data: MetricHistory[]; colors: string[]; height?: number
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || data.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = containerRef.current.clientWidth;
    const margin = { top: 20, right: 20, bottom: 30, left: 40 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.attr("width", width).attr("height", height);

    const g = svg.append("g")
      .attr("transform", `translate(${margin.left},${margin.top})`);

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(data, d => new Date(d.timestamp)) as [Date, Date])
      .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([innerHeight, 0]);

    // Grid lines
    g.append("g")
      .attr("class", "grid")
      .attr("opacity", 0.1)
      .call(d3.axisLeft(yScale).tickSize(-innerWidth).tickFormat(() => ""));

    // X Axis
    g.append("g")
      .attr("transform", `translate(0,${innerHeight})`)
      .attr("color", "#4b5563")
      .call(d3.axisBottom(xScale).ticks(5).tickFormat(d => d3.timeFormat("%H:%M")(d as Date)));

    // Y Axis
    g.append("g")
      .attr("color", "#4b5563")
      .call(d3.axisLeft(yScale).ticks(5).tickFormat(d => `${d}%`));

    // Lines for each GPU
    if (data.length > 0 && data[0].values) {
      const numGPUs = data[0].values.length;

      for (let i = 0; i < numGPUs; i++) {
        const line = d3.line<MetricHistory>()
          .x(d => xScale(new Date(d.timestamp)))
          .y(d => yScale(d.values[i] || 0))
          .curve(d3.curveMonotoneX);

        // Gradient area
        const area = d3.area<MetricHistory>()
          .x(d => xScale(new Date(d.timestamp)))
          .y0(innerHeight)
          .y1(d => yScale(d.values[i] || 0))
          .curve(d3.curveMonotoneX);

        const areaGradient = svg.append("defs")
          .append("linearGradient")
          .attr("id", `area-gradient-${i}`)
          .attr("x1", "0%").attr("y1", "0%")
          .attr("x2", "0%").attr("y2", "100%");
        areaGradient.append("stop")
          .attr("offset", "0%")
          .attr("stop-color", colors[i % colors.length])
          .attr("stop-opacity", 0.3);
        areaGradient.append("stop")
          .attr("offset", "100%")
          .attr("stop-color", colors[i % colors.length])
          .attr("stop-opacity", 0);

        g.append("path")
          .datum(data)
          .attr("fill", `url(#area-gradient-${i})`)
          .attr("d", area);

        g.append("path")
          .datum(data)
          .attr("fill", "none")
          .attr("stroke", colors[i % colors.length])
          .attr("stroke-width", 2)
          .attr("d", line);
      }
    }

  }, [data, colors, height]);

  return (
    <div ref={containerRef} className="w-full">
      <svg ref={svgRef} />
    </div>
  );
}

// Donut Chart for Task Status
function DonutChart({ data }: { data: { label: string; value: number; color: string }[] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    const width = 200;
    const height = 200;
    const radius = Math.min(width, height) / 2;

    const g = svg
      .attr("width", width)
      .attr("height", height)
      .append("g")
      .attr("transform", `translate(${width / 2}, ${height / 2})`);

    const pie = d3.pie<{ label: string; value: number; color: string }>()
      .value(d => d.value)
      .sort(null);

    const arc = d3.arc<d3.PieArcDatum<{ label: string; value: number; color: string }>>()
      .innerRadius(radius * 0.6)
      .outerRadius(radius * 0.9)
      .cornerRadius(4)
      .padAngle(0.02);

    const total = data.reduce((sum, d) => sum + d.value, 0);

    g.selectAll("path")
      .data(pie(data))
      .enter()
      .append("path")
      .attr("d", arc)
      .attr("fill", d => d.data.color)
      .attr("opacity", 0.9)
      .transition()
      .duration(750)
      .attrTween("d", function(d) {
        const interpolate = d3.interpolate({ startAngle: 0, endAngle: 0 }, d);
        return function(t) {
          return arc(interpolate(t)) || "";
        };
      });

    // Center text
    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "-0.2em")
      .attr("fill", "#fff")
      .attr("font-size", "28px")
      .attr("font-weight", "bold")
      .text(total);

    g.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", "1.2em")
      .attr("fill", "#9ca3af")
      .attr("font-size", "12px")
      .text("Total Tasks");

  }, [data]);

  return <svg ref={svgRef} />;
}

export default function Dashboard() {
  const [connected, setConnected] = useState(false);
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [selectedModel, setSelectedModel] = useState<Model | null>(null);
  const [liveOutput, setLiveOutput] = useState<{ [key: string]: string[] }>({});
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [entryPoint, setEntryPoint] = useState("main.py");
  const [activeTab, setActiveTab] = useState<"code" | "file">("code");
  const [mainTab, setMainTab] = useState<"dashboard" | "models">("dashboard");
  const [metricHistory, setMetricHistory] = useState<MetricHistory[]>([]);
  const [showModelRegister, setShowModelRegister] = useState(false);
  const [showInferenceModal, setShowInferenceModal] = useState(false);
  const [inferenceModel, setInferenceModel] = useState<Model | null>(null);
  const [inferenceInput, setInferenceInput] = useState("{}");
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [modelName, setModelName] = useState("");
  const [modelType, setModelType] = useState("pytorch");
  const wsRef = useRef<WebSocket | null>(null);
  const outputRef = useRef<HTMLPreElement>(null);

  const gpuColors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b"];

  const getWsUrl = () => {
    if (typeof window === "undefined") return "ws://localhost:9825/ws";
    const host = window.location.hostname;
    return `ws://${host}:9825/ws`;
  };

  const getApiUrl = () => {
    if (typeof window === "undefined") return "http://localhost:9825";
    const host = window.location.hostname;
    return `http://${host}:9825`;
  };

  const fetchModels = useCallback(async () => {
    try {
      const res = await fetch(`${getApiUrl()}/models`);
      if (res.ok) {
        const data = await res.json();
        setModels(data);
      }
    } catch {
      console.error("Failed to fetch models");
    }
  }, []);

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(getWsUrl());

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onclose = () => {
      setConnected(false);
      setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = () => {
      setConnected(false);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      switch (data.type) {
        case "init":
          setGpus(data.gpus || []);
          setTasks(data.tasks || []);
          break;

        case "gpu_update":
          setGpus(data.gpus || []);
          // Record history
          if (data.gpus && data.gpus.length > 0) {
            setMetricHistory(prev => {
              const newEntry = {
                timestamp: Date.now(),
                values: data.gpus.map((g: GPU) => g.utilization_percent)
              };
              const updated = [...prev, newEntry].slice(-60); // Keep last 60 points
              return updated;
            });
          }
          break;

        case "task_update":
          setTasks((prev) => {
            const idx = prev.findIndex((t) => t.task_id === data.task.task_id);
            if (idx >= 0) {
              const updated = [...prev];
              updated[idx] = data.task;
              return updated;
            }
            return [data.task, ...prev];
          });
          if (selectedTask?.task_id === data.task.task_id) {
            setSelectedTask(data.task);
          }
          break;

        case "task_output":
          setLiveOutput((prev) => ({
            ...prev,
            [data.task_id]: [...(prev[data.task_id] || []), data.line],
          }));
          break;
      }
    };

    wsRef.current = ws;
  }, [selectedTask?.task_id]);

  useEffect(() => {
    connectWebSocket();
    fetchModels();
    return () => {
      wsRef.current?.close();
    };
  }, [connectWebSocket, fetchModels]);

  useEffect(() => {
    if (mainTab === "models") {
      fetchModels();
    }
  }, [mainTab, fetchModels]);

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [liveOutput, selectedTask]);

  const submitCode = async () => {
    if (!code.trim()) return;
    setIsSubmitting(true);
    try {
      await fetch(`${getApiUrl()}/run/async`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code, timeout: 3600 }),
      });
      setCode("");
    } catch {
      alert("Failed to submit task");
    }
    setIsSubmitting(false);
  };

  const submitFile = async () => {
    if (!uploadFile) return;
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("file", uploadFile);
      formData.append("entry_point", entryPoint);
      formData.append("timeout", "3600");

      await fetch(`${getApiUrl()}/run/project`, {
        method: "POST",
        body: formData,
      });
      setUploadFile(null);
    } catch {
      alert("Failed to upload file");
    }
    setIsSubmitting(false);
  };

  const registerModel = async () => {
    if (!modelFile || !modelName.trim()) return;
    setIsSubmitting(true);
    try {
      const formData = new FormData();
      formData.append("model_file", modelFile);
      formData.append("name", modelName);
      formData.append("model_type", modelType);
      formData.append("framework", "pytorch");

      const res = await fetch(`${getApiUrl()}/model/register`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        setShowModelRegister(false);
        setModelFile(null);
        setModelName("");
        fetchModels();
      } else {
        alert("Failed to register model");
      }
    } catch {
      alert("Failed to register model");
    }
    setIsSubmitting(false);
  };

  const deleteModel = async (modelId: string) => {
    if (!confirm("Are you sure you want to delete this model?")) return;
    try {
      const res = await fetch(`${getApiUrl()}/model/${modelId}`, {
        method: "DELETE",
      });
      if (res.ok) {
        fetchModels();
        setSelectedModel(null);
      } else {
        alert("Failed to delete model");
      }
    } catch {
      alert("Failed to delete model");
    }
  };

  const runInference = async () => {
    if (!inferenceModel) return;
    setIsSubmitting(true);
    try {
      let inputData;
      try {
        inputData = JSON.parse(inferenceInput);
      } catch {
        alert("Invalid JSON input");
        setIsSubmitting(false);
        return;
      }

      const res = await fetch(`${getApiUrl()}/model/${inferenceModel.model_id}/inference`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input_data: inputData, timeout: 60 }),
      });

      if (res.ok) {
        const data = await res.json();
        alert(`Inference task started: ${data.task_id}`);
        setShowInferenceModal(false);
        setInferenceInput("{}");
        setMainTab("dashboard");
      } else {
        alert("Failed to start inference");
      }
    } catch {
      alert("Failed to start inference");
    }
    setIsSubmitting(false);
  };

  const downloadModel = (modelId: string) => {
    window.open(`${getApiUrl()}/model/${modelId}/download`, "_blank");
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed": return "bg-emerald-500";
      case "running": return "bg-blue-500";
      case "queued": return "bg-amber-500";
      case "pending": return "bg-slate-400";
      case "failed": return "bg-rose-500";
      case "ready": return "bg-emerald-500";
      case "archived": return "bg-slate-500";
      default: return "bg-slate-400";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "completed": return "Completed";
      case "running": return "Running";
      case "queued": return "Queued";
      case "pending": return "Pending";
      case "failed": return "Failed";
      case "ready": return "Ready";
      case "archived": return "Archived";
      default: return status;
    }
  };

  const formatTime = (dateStr: string | null) => {
    if (!dateStr) return "-";
    return new Date(dateStr).toLocaleTimeString("ko-KR");
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return "-";
    return new Date(dateStr).toLocaleDateString("ko-KR", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  const formatDuration = (start: string | null, end: string | null) => {
    if (!start) return "-";
    const startTime = new Date(start).getTime();
    const endTime = end ? new Date(end).getTime() : Date.now();
    const seconds = Math.floor((endTime - startTime) / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${seconds % 60}s`;
  };

  const formatFileSize = (bytes: number | null) => {
    if (!bytes) return "-";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const getLiveOutputForTask = (taskId: string) => {
    return liveOutput[taskId]?.join("") || "";
  };

  const getTaskStats = () => {
    const stats = {
      completed: tasks.filter(t => t.status === "completed").length,
      running: tasks.filter(t => t.status === "running").length,
      queued: tasks.filter(t => t.status === "queued").length,
      failed: tasks.filter(t => t.status === "failed").length,
    };
    return [
      { label: "Completed", value: stats.completed, color: "#10b981" },
      { label: "Running", value: stats.running, color: "#3b82f6" },
      { label: "Queued", value: stats.queued, color: "#f59e0b" },
      { label: "Failed", value: stats.failed, color: "#ef4444" },
    ].filter(d => d.value > 0);
  };

  const usedMemory = gpus.reduce((sum, g) => sum + g.memory_used_mb, 0);
  const avgUtilization = gpus.length > 0
    ? Math.round(gpus.reduce((sum, g) => sum + g.utilization_percent, 0) / gpus.length)
    : 0;
  const busyGPUs = gpus.filter(g => g.task_id).length;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950 text-white">
      {/* Header */}
      <header className="border-b border-slate-800/50 backdrop-blur-xl bg-slate-900/70 sticky top-0 z-40 transition-all duration-300">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-lg animate-gradient-flow shadow-lg shadow-blue-500/20 hover:shadow-blue-500/40 transition-shadow duration-300">
                V
              </div>
              <div>
                <h1 className="text-xl font-bold bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                  VrunGPU Dashboard
                </h1>
                <p className="text-xs text-slate-500">Remote GPU Execution Server v0.4.0</p>
              </div>
            </div>

            {/* Main Navigation */}
            <div className="flex items-center gap-2 bg-slate-800/50 rounded-xl p-1">
              <button
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 ${
                  mainTab === "dashboard"
                    ? "bg-blue-500 text-white shadow-lg shadow-blue-500/25"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                }`}
                onClick={() => setMainTab("dashboard")}
              >
                Dashboard
              </button>
              <button
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 flex items-center gap-2 ${
                  mainTab === "models"
                    ? "bg-purple-500 text-white shadow-lg shadow-purple-500/25"
                    : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                }`}
                onClick={() => setMainTab("models")}
              >
                Models
                {models.length > 0 && (
                  <span className="px-1.5 py-0.5 rounded-md text-xs bg-purple-600/50">{models.length}</span>
                )}
              </button>
            </div>

            <div className="flex items-center gap-6">
              <div className="text-right text-sm">
                <p className="text-slate-400">API Endpoint</p>
                <p className="font-mono text-xs text-slate-500">{getApiUrl()}</p>
              </div>
              <div className={`flex items-center gap-2 px-4 py-2 rounded-full border transition-all duration-500 ${
                connected
                  ? "bg-emerald-500/10 border-emerald-500/30 shadow-lg shadow-emerald-500/10"
                  : "bg-rose-500/10 border-rose-500/30"
              }`}>
                <span className={`w-2 h-2 rounded-full transition-all duration-300 ${connected ? "bg-emerald-500 animate-pulse shadow-lg shadow-emerald-500/50" : "bg-rose-500"}`} />
                <span className={`text-sm transition-colors duration-300 ${connected ? "text-emerald-400" : "text-rose-400"}`}>
                  {connected ? "Connected" : "Disconnected"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8">
        {mainTab === "dashboard" ? (
          <>
            {/* Stats Overview */}
            <div className="grid grid-cols-4 gap-4 mb-8">
              {[
                { label: "Total GPUs", value: gpus.length, icon: "GPU", color: "from-blue-500 to-cyan-500" },
                { label: "Active Tasks", value: busyGPUs, icon: "RUN", color: "from-emerald-500 to-teal-500" },
                { label: "Avg Utilization", value: `${avgUtilization}%`, icon: "CPU", color: "from-purple-500 to-pink-500" },
                { label: "Memory Used", value: `${Math.round(usedMemory / 1024)}GB`, icon: "MEM", color: "from-amber-500 to-orange-500" },
              ].map((stat, idx) => (
                <div
                  key={stat.label}
                  className={`relative overflow-hidden rounded-2xl bg-slate-800/30 border border-slate-700/50 p-5 hover-lift hover-glow opacity-0 animate-fade-in-up stagger-${idx + 1}`}
                  style={{ animationFillMode: 'forwards' }}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-slate-400 text-sm mb-1">{stat.label}</p>
                      <p className="text-3xl font-bold transition-all duration-500">{stat.value}</p>
                    </div>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center text-xs font-bold opacity-80 animate-float`}>
                      {stat.icon}
                    </div>
                  </div>
                  <div className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.color} opacity-50 transition-all duration-300`} />
                  <div className={`absolute inset-0 bg-gradient-to-r ${stat.color} opacity-0 hover:opacity-5 transition-opacity duration-300`} />
                </div>
              ))}
            </div>

            <div className="grid grid-cols-12 gap-6">
              {/* GPU Cards */}
              <div className="col-span-8">
                <div className="rounded-2xl glass p-6 opacity-0 animate-fade-in-up stagger-2" style={{ animationFillMode: 'forwards' }}>
                  <h2 className="text-lg font-semibold mb-6 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-blue-500 animate-pulse" />
                    GPU Status
                  </h2>
                  <div className="grid grid-cols-2 gap-4">
                    {gpus.map((gpu, idx) => (
                      <div
                        key={gpu.index}
                        className={`rounded-xl p-5 transition-all duration-500 ease-out transform hover:scale-[1.02] ${
                          gpu.task_id
                            ? "bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-2 border-blue-500/30 shadow-lg shadow-blue-500/10 animate-pulse-glow"
                            : "bg-slate-800/50 border border-slate-700/50 hover:border-slate-600/50 hover:bg-slate-800/70"
                        }`}
                      >
                        <div className="flex justify-between items-start mb-4">
                          <div>
                            <div className="flex items-center gap-2 mb-1">
                              <span className={`w-2 h-2 rounded-full transition-all duration-300 ${gpu.task_id ? "bg-blue-500 shadow-lg shadow-blue-500/50" : "bg-emerald-500"}`}>
                                {gpu.task_id && <span className="absolute inset-0 rounded-full bg-blue-500 animate-ping opacity-75" />}
                              </span>
                              <h3 className="font-semibold">GPU {gpu.index}</h3>
                            </div>
                            <p className="text-xs text-slate-400 truncate max-w-[180px]">{gpu.name}</p>
                          </div>
                          <span className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-300 ${
                            gpu.task_id ? "bg-blue-500/20 text-blue-400" : "bg-emerald-500/20 text-emerald-400"
                          }`}>
                            {gpu.task_id ? "Busy" : "Idle"}
                          </span>
                        </div>

                        <div className="flex items-center justify-between">
                          <GaugeChart
                            value={gpu.utilization_percent}
                            max={100}
                            label="Utilization"
                            color={gpuColors[idx % gpuColors.length]}
                            size={100}
                          />
                          <GaugeChart
                            value={gpu.memory_used_mb}
                            max={gpu.memory_total_mb}
                            label="Memory"
                            color={gpuColors[(idx + 1) % gpuColors.length]}
                            size={100}
                          />
                        </div>

                        <div className={`overflow-hidden transition-all duration-500 ease-out ${gpu.task_id ? "max-h-20 opacity-100 mt-3 pt-3 border-t border-slate-700/50" : "max-h-0 opacity-0"}`}>
                          <p className="text-xs text-slate-400">
                            Running: <span className="text-blue-400 font-mono">{gpu.task_id?.slice(0, 12)}...</span>
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* GPU History Chart */}
                <div className="rounded-2xl glass p-6 mt-6 opacity-0 animate-fade-in-up stagger-3" style={{ animationFillMode: 'forwards' }}>
                  <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-purple-500 animate-breathe" />
                    GPU Utilization History
                  </h2>
                  <div className="flex items-center gap-4 mb-4">
                    {gpus.map((gpu, idx) => (
                      <div key={gpu.index} className="flex items-center gap-2 text-sm group cursor-default">
                        <span
                          className="w-3 h-3 rounded transition-transform duration-200 group-hover:scale-125"
                          style={{ backgroundColor: gpuColors[idx % gpuColors.length] }}
                        />
                        <span className="text-slate-400 group-hover:text-slate-200 transition-colors duration-200">GPU {gpu.index}</span>
                      </div>
                    ))}
                  </div>
                  {metricHistory.length > 1 ? (
                    <LineChart data={metricHistory} colors={gpuColors} height={180} />
                  ) : (
                    <div className="h-[180px] flex items-center justify-center text-slate-500">
                      <div className="flex items-center gap-2">
                        <span className="w-2 h-2 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '0ms' }} />
                        <span className="w-2 h-2 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '150ms' }} />
                        <span className="w-2 h-2 rounded-full bg-slate-500 animate-bounce" style={{ animationDelay: '300ms' }} />
                        <span className="ml-2">Collecting data...</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Right Sidebar */}
              <div className="col-span-4 space-y-6">
                {/* Task Stats */}
                <div className="rounded-2xl glass p-6 opacity-0 animate-slide-in-right stagger-2" style={{ animationFillMode: 'forwards' }}>
                  <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-amber-500 animate-breathe" />
                    Task Statistics
                  </h2>
                  <div className="flex justify-center">
                    {tasks.length > 0 ? (
                      <DonutChart data={getTaskStats()} />
                    ) : (
                      <div className="h-[200px] flex items-center justify-center text-slate-500">
                        No tasks yet
                      </div>
                    )}
                  </div>
                  <div className="grid grid-cols-2 gap-2 mt-4">
                    {getTaskStats().map((stat, idx) => (
                      <div
                        key={stat.label}
                        className="flex items-center gap-2 text-sm opacity-0 animate-fade-in-up"
                        style={{ animationDelay: `${idx * 100}ms`, animationFillMode: 'forwards' }}
                      >
                        <span className="w-3 h-3 rounded transition-transform duration-200 hover:scale-125" style={{ backgroundColor: stat.color }} />
                        <span className="text-slate-400">{stat.label}:</span>
                        <span className="font-semibold">{stat.value}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Submit Task */}
                <div className="rounded-2xl glass p-6 opacity-0 animate-slide-in-right stagger-3" style={{ animationFillMode: 'forwards' }}>
                  <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    Submit Task
                  </h2>

                  <div className="flex mb-4 bg-slate-800/50 rounded-lg p-1">
                    <button
                      className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                        activeTab === "code"
                          ? "bg-blue-500 text-white shadow-lg shadow-blue-500/25"
                          : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                      }`}
                      onClick={() => setActiveTab("code")}
                    >
                      Code
                    </button>
                    <button
                      className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-all duration-300 ${
                        activeTab === "file"
                          ? "bg-blue-500 text-white shadow-lg shadow-blue-500/25"
                          : "text-slate-400 hover:text-white hover:bg-slate-700/50"
                      }`}
                      onClick={() => setActiveTab("file")}
                    >
                      Upload
                    </button>
                  </div>

                  <div className="relative overflow-hidden">
                    <div className={`transition-all duration-300 ease-out ${activeTab === "code" ? "opacity-100 translate-x-0" : "opacity-0 -translate-x-full absolute inset-0"}`}>
                      <textarea
                        className="w-full h-32 bg-slate-900/50 text-emerald-400 font-mono text-sm p-4 rounded-xl border border-slate-700/50 focus:border-blue-500/50 focus:outline-none resize-none placeholder:text-slate-600 transition-all duration-300 focus:shadow-lg focus:shadow-blue-500/10"
                        placeholder="# Python code..."
                        value={code}
                        onChange={(e) => setCode(e.target.value)}
                      />
                      <button
                        className="w-full mt-3 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:from-slate-600 disabled:to-slate-700 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-blue-500/25 disabled:shadow-none hover:shadow-xl hover:shadow-blue-500/30 active:scale-[0.98]"
                        onClick={submitCode}
                        disabled={isSubmitting || !code.trim()}
                      >
                        {isSubmitting ? (
                          <span className="flex items-center justify-center gap-2">
                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Submitting...
                          </span>
                        ) : "Execute"}
                      </button>
                    </div>
                    <div className={`transition-all duration-300 ease-out ${activeTab === "file" ? "opacity-100 translate-x-0" : "opacity-0 translate-x-full absolute inset-0"}`}>
                      <div className="border-2 border-dashed border-slate-600/50 hover:border-blue-500/50 rounded-xl p-6 text-center transition-all duration-300 cursor-pointer hover:bg-slate-800/30">
                        <input
                          type="file"
                          accept=".py,.zip"
                          className="hidden"
                          id="file-upload"
                          onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                        />
                        <label htmlFor="file-upload" className="cursor-pointer block">
                          {uploadFile ? (
                            <div className="animate-fade-in-scale">
                              <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-blue-500/20 flex items-center justify-center animate-float">
                                <span className="text-blue-400 text-xl">📦</span>
                              </div>
                              <p className="text-blue-400 font-medium">{uploadFile.name}</p>
                              <p className="text-xs text-slate-500 mt-1">
                                {(uploadFile.size / 1024).toFixed(1)} KB
                              </p>
                            </div>
                          ) : (
                            <div>
                              <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-slate-700/50 flex items-center justify-center transition-transform duration-300 hover:scale-110">
                                <span className="text-slate-400 text-xl">📁</span>
                              </div>
                              <p className="text-slate-400">Drop file or click to select</p>
                              <p className="text-xs text-slate-600 mt-1">.py or .zip</p>
                            </div>
                          )}
                        </label>
                      </div>
                      <input
                        type="text"
                        className="w-full bg-slate-900/50 text-white text-sm p-3 rounded-xl border border-slate-700/50 focus:border-blue-500/50 focus:outline-none mt-3 placeholder:text-slate-600 transition-all duration-300 focus:shadow-lg focus:shadow-blue-500/10"
                        placeholder="Entry point (e.g., train.py)"
                        value={entryPoint}
                        onChange={(e) => setEntryPoint(e.target.value)}
                      />
                      <button
                        className="w-full mt-3 bg-gradient-to-r from-emerald-500 to-teal-600 hover:from-emerald-600 hover:to-teal-700 disabled:from-slate-600 disabled:to-slate-700 py-3 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-emerald-500/25 disabled:shadow-none hover:shadow-xl hover:shadow-emerald-500/30 active:scale-[0.98]"
                        onClick={submitFile}
                        disabled={isSubmitting || !uploadFile}
                      >
                        {isSubmitting ? (
                          <span className="flex items-center justify-center gap-2">
                            <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            Uploading...
                          </span>
                        ) : "Upload & Execute"}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Task List */}
            <div className="rounded-2xl glass p-6 mt-6 opacity-0 animate-fade-in-up stagger-4" style={{ animationFillMode: 'forwards' }}>
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-rose-500" />
                Task History
              </h2>
              <div className="overflow-hidden rounded-xl border border-slate-700/50">
                <table className="w-full">
                  <thead className="bg-slate-800/50">
                    <tr>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Task ID</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Type</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Status</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Progress</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">GPU</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Duration</th>
                      <th className="px-5 py-4 text-left text-xs font-medium text-slate-400 uppercase tracking-wider">Result</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/30">
                    {tasks.length === 0 ? (
                      <tr>
                        <td colSpan={7} className="px-5 py-12 text-center text-slate-500">
                          No tasks submitted yet
                        </td>
                      </tr>
                    ) : (
                      tasks.slice(0, 10).map((task, idx) => (
                        <tr
                          key={task.task_id}
                          className="hover:bg-slate-700/30 cursor-pointer transition-all duration-200 hover:shadow-lg opacity-0 animate-fade-in-up"
                          style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'forwards' }}
                          onClick={() => setSelectedTask(task)}
                        >
                          <td className="px-5 py-4">
                            <span className="font-mono text-sm text-slate-300">{task.task_id.slice(0, 12)}...</span>
                          </td>
                          <td className="px-5 py-4">
                            <span className={`px-2 py-1 rounded-md text-xs font-medium ${
                              task.task_type === "inference"
                                ? "bg-purple-500/20 text-purple-400"
                                : "bg-blue-500/20 text-blue-400"
                            }`}>
                              {task.task_type || "training"}
                            </span>
                          </td>
                          <td className="px-5 py-4">
                            <span className="inline-flex items-center gap-2">
                              <span className={`w-2 h-2 rounded-full transition-all duration-300 ${getStatusColor(task.status)} ${task.status === "running" ? "shadow-lg shadow-blue-500/50" : ""}`}>
                                {task.status === "running" && <span className="absolute inset-0 rounded-full bg-blue-500 animate-ping opacity-50" />}
                              </span>
                              <span className="text-sm">{getStatusText(task.status)}</span>
                            </span>
                          </td>
                          <td className="px-5 py-4">
                            {task.progress !== undefined && task.progress !== null ? (
                              <div className="flex items-center gap-2">
                                <div className="w-16 h-2 bg-slate-700 rounded-full overflow-hidden">
                                  <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
                                    style={{ width: `${task.progress}%` }}
                                  />
                                </div>
                                <span className="text-xs text-slate-400">{Math.round(task.progress)}%</span>
                              </div>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                          <td className="px-5 py-4 text-sm text-slate-400">
                            {task.gpu_id !== null ? (
                              <span className="px-2 py-1 rounded-md bg-slate-700/50 text-xs transition-all duration-200 hover:bg-slate-600/50">GPU {task.gpu_id}</span>
                            ) : "-"}
                          </td>
                          <td className="px-5 py-4 text-sm text-slate-400">
                            {formatDuration(task.started_at, task.completed_at)}
                          </td>
                          <td className="px-5 py-4 text-sm">
                            {task.return_code !== null ? (
                              <span className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-all duration-200 ${
                                task.return_code === 0
                                  ? "bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30"
                                  : "bg-rose-500/20 text-rose-400 hover:bg-rose-500/30"
                              }`}>
                                {task.return_code === 0 ? "Success" : `Error (${task.return_code})`}
                              </span>
                            ) : task.status === "running" ? (
                              <span className="text-blue-400 text-xs flex items-center gap-1">
                                <span className="w-1.5 h-1.5 rounded-full bg-blue-400 animate-pulse" />
                                Processing...
                              </span>
                            ) : (
                              <span className="text-slate-600">-</span>
                            )}
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </>
        ) : (
          /* Models Tab */
          <div className="space-y-6">
            {/* Models Header */}
            <div className="flex justify-between items-center">
              <div>
                <h2 className="text-2xl font-bold">Model Management</h2>
                <p className="text-slate-400 text-sm mt-1">Register, manage, and run inference with your models</p>
              </div>
              <button
                className="px-4 py-2.5 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 rounded-xl font-medium transition-all duration-300 shadow-lg shadow-purple-500/25 hover:shadow-xl hover:shadow-purple-500/30 active:scale-[0.98] flex items-center gap-2"
                onClick={() => setShowModelRegister(true)}
              >
                <span>+</span>
                Register Model
              </button>
            </div>

            {/* Models Stats */}
            <div className="grid grid-cols-4 gap-4">
              {[
                { label: "Total Models", value: models.length, icon: "📦", color: "from-purple-500 to-pink-500" },
                { label: "Ready", value: models.filter(m => m.status === "ready").length, icon: "✓", color: "from-emerald-500 to-teal-500" },
                { label: "Total Size", value: formatFileSize(models.reduce((sum, m) => sum + (m.file_size || 0), 0)), icon: "💾", color: "from-blue-500 to-cyan-500" },
                { label: "Inference Tasks", value: tasks.filter(t => t.task_type === "inference").length, icon: "🔮", color: "from-amber-500 to-orange-500" },
              ].map((stat, idx) => (
                <div
                  key={stat.label}
                  className={`relative overflow-hidden rounded-2xl bg-slate-800/30 border border-slate-700/50 p-5 hover-lift hover-glow opacity-0 animate-fade-in-up stagger-${idx + 1}`}
                  style={{ animationFillMode: 'forwards' }}
                >
                  <div className="flex justify-between items-start">
                    <div>
                      <p className="text-slate-400 text-sm mb-1">{stat.label}</p>
                      <p className="text-3xl font-bold">{stat.value}</p>
                    </div>
                    <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${stat.color} flex items-center justify-center text-xl opacity-80`}>
                      {stat.icon}
                    </div>
                  </div>
                  <div className={`absolute bottom-0 left-0 right-0 h-1 bg-gradient-to-r ${stat.color} opacity-50`} />
                </div>
              ))}
            </div>

            {/* Models Grid */}
            <div className="rounded-2xl glass p-6">
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-purple-500" />
                Registered Models
              </h3>

              {models.length === 0 ? (
                <div className="text-center py-16">
                  <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-slate-800/50 flex items-center justify-center text-4xl">
                    📦
                  </div>
                  <h3 className="text-lg font-medium text-slate-300 mb-2">No models registered</h3>
                  <p className="text-slate-500 text-sm mb-4">Upload your first model to get started with inference</p>
                  <button
                    className="px-4 py-2 bg-purple-500/20 text-purple-400 rounded-lg hover:bg-purple-500/30 transition-colors"
                    onClick={() => setShowModelRegister(true)}
                  >
                    Register Model
                  </button>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-4">
                  {models.map((model, idx) => (
                    <div
                      key={model.model_id}
                      className="rounded-xl bg-slate-800/50 border border-slate-700/50 p-5 hover:border-purple-500/50 transition-all duration-300 cursor-pointer hover:shadow-lg hover:shadow-purple-500/10 opacity-0 animate-fade-in-up"
                      style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'forwards' }}
                      onClick={() => setSelectedModel(model)}
                    >
                      <div className="flex justify-between items-start mb-3">
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center text-lg">
                            🧠
                          </div>
                          <div>
                            <h4 className="font-medium text-white truncate max-w-[150px]">{model.name}</h4>
                            <p className="text-xs text-slate-500 font-mono">{model.model_id}</p>
                          </div>
                        </div>
                        <span className={`px-2 py-1 rounded-md text-xs ${
                          model.status === "ready"
                            ? "bg-emerald-500/20 text-emerald-400"
                            : "bg-slate-500/20 text-slate-400"
                        }`}>
                          {model.status}
                        </span>
                      </div>

                      <div className="space-y-2 text-sm">
                        <div className="flex justify-between">
                          <span className="text-slate-500">Type</span>
                          <span className="text-slate-300">{model.model_type || "general"}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Framework</span>
                          <span className="text-slate-300">{model.framework}</span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-500">Size</span>
                          <span className="text-slate-300">{formatFileSize(model.file_size)}</span>
                        </div>
                      </div>

                      <div className="flex gap-2 mt-4 pt-4 border-t border-slate-700/50">
                        <button
                          className="flex-1 px-3 py-2 bg-purple-500/20 text-purple-400 rounded-lg text-sm hover:bg-purple-500/30 transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            setInferenceModel(model);
                            setShowInferenceModal(true);
                          }}
                        >
                          Run Inference
                        </button>
                        <button
                          className="px-3 py-2 bg-slate-700/50 text-slate-400 rounded-lg text-sm hover:bg-slate-600/50 transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            downloadModel(model.model_id);
                          }}
                        >
                          ↓
                        </button>
                        <button
                          className="px-3 py-2 bg-rose-500/20 text-rose-400 rounded-lg text-sm hover:bg-rose-500/30 transition-colors"
                          onClick={(e) => {
                            e.stopPropagation();
                            deleteModel(model.model_id);
                          }}
                        >
                          ×
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Recent Inference Tasks */}
            <div className="rounded-2xl glass p-6">
              <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-amber-500" />
                Recent Inference Tasks
              </h3>
              <div className="overflow-hidden rounded-xl border border-slate-700/50">
                <table className="w-full">
                  <thead className="bg-slate-800/50">
                    <tr>
                      <th className="px-5 py-3 text-left text-xs font-medium text-slate-400 uppercase">Task ID</th>
                      <th className="px-5 py-3 text-left text-xs font-medium text-slate-400 uppercase">Model</th>
                      <th className="px-5 py-3 text-left text-xs font-medium text-slate-400 uppercase">Status</th>
                      <th className="px-5 py-3 text-left text-xs font-medium text-slate-400 uppercase">Duration</th>
                      <th className="px-5 py-3 text-left text-xs font-medium text-slate-400 uppercase">Created</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-700/30">
                    {tasks.filter(t => t.task_type === "inference").length === 0 ? (
                      <tr>
                        <td colSpan={5} className="px-5 py-8 text-center text-slate-500">
                          No inference tasks yet
                        </td>
                      </tr>
                    ) : (
                      tasks.filter(t => t.task_type === "inference").slice(0, 5).map((task) => (
                        <tr
                          key={task.task_id}
                          className="hover:bg-slate-700/30 cursor-pointer transition-colors"
                          onClick={() => setSelectedTask(task)}
                        >
                          <td className="px-5 py-3 font-mono text-sm text-slate-300">{task.task_id.slice(0, 12)}...</td>
                          <td className="px-5 py-3 text-sm">
                            {task.model_id ? (
                              <span className="text-purple-400">{models.find(m => m.model_id === task.model_id)?.name || task.model_id}</span>
                            ) : "-"}
                          </td>
                          <td className="px-5 py-3">
                            <span className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-md text-xs ${
                              task.status === "completed" ? "bg-emerald-500/20 text-emerald-400" :
                              task.status === "running" ? "bg-blue-500/20 text-blue-400" :
                              task.status === "failed" ? "bg-rose-500/20 text-rose-400" :
                              "bg-slate-500/20 text-slate-400"
                            }`}>
                              <span className={`w-1.5 h-1.5 rounded-full ${getStatusColor(task.status)}`} />
                              {getStatusText(task.status)}
                            </span>
                          </td>
                          <td className="px-5 py-3 text-sm text-slate-400">{formatDuration(task.started_at, task.completed_at)}</td>
                          <td className="px-5 py-3 text-sm text-slate-500">{formatTime(task.created_at)}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Task Detail Modal */}
      {selectedTask && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 animate-backdrop-in"
          onClick={() => setSelectedTask(null)}
        >
          <div
            className="bg-slate-900 rounded-2xl max-w-4xl w-full max-h-[85vh] overflow-hidden flex flex-col border border-slate-700/50 shadow-2xl animate-modal-in"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/50">
              <div className="flex items-center gap-4">
                <h3 className="font-semibold text-lg">Task Details</h3>
                <span className={`px-2 py-1 rounded-md text-xs ${
                  selectedTask.task_type === "inference"
                    ? "bg-purple-500/20 text-purple-400"
                    : "bg-blue-500/20 text-blue-400"
                }`}>
                  {selectedTask.task_type || "training"}
                </span>
                <span className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs bg-slate-700/50 transition-all duration-300">
                  <span className={`w-2 h-2 rounded-full transition-all duration-300 ${getStatusColor(selectedTask.status)} ${selectedTask.status === "running" ? "shadow-lg shadow-blue-500/50" : ""}`} />
                  {getStatusText(selectedTask.status)}
                </span>
              </div>
              <button
                className="w-8 h-8 rounded-lg bg-slate-700/50 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-all duration-200 hover:rotate-90"
                onClick={() => setSelectedTask(null)}
              >
                ✕
              </button>
            </div>

            <div className="p-6 overflow-y-auto flex-1">
              <div className="grid grid-cols-4 gap-4 mb-6">
                {[
                  { label: "Task ID", value: selectedTask.task_id.slice(0, 16) + "..." },
                  { label: "GPU", value: selectedTask.gpu_id !== null ? `GPU ${selectedTask.gpu_id}` : "-" },
                  { label: "Duration", value: formatDuration(selectedTask.started_at, selectedTask.completed_at) },
                  { label: "Return Code", value: selectedTask.return_code ?? "-" },
                ].map((item, idx) => (
                  <div
                    key={item.label}
                    className="bg-slate-800/50 rounded-xl p-4 opacity-0 animate-fade-in-up hover:bg-slate-800/70 transition-colors duration-200"
                    style={{ animationDelay: `${idx * 50}ms`, animationFillMode: 'forwards' }}
                  >
                    <p className="text-xs text-slate-400 mb-1">{item.label}</p>
                    <p className="font-mono text-sm truncate">{item.value}</p>
                  </div>
                ))}
              </div>

              {/* Progress Bar */}
              {selectedTask.progress !== undefined && selectedTask.progress !== null && (
                <div className="mb-6 opacity-0 animate-fade-in-up stagger-1" style={{ animationFillMode: 'forwards' }}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium">Progress</span>
                    <span className="text-sm text-slate-400">{Math.round(selectedTask.progress)}%</span>
                  </div>
                  <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500 rounded-full"
                      style={{ width: `${selectedTask.progress}%` }}
                    />
                  </div>
                  {selectedTask.progress_message && (
                    <p className="text-xs text-slate-500 mt-2">{selectedTask.progress_message}</p>
                  )}
                </div>
              )}

              <div className="mb-4 opacity-0 animate-fade-in-up stagger-2" style={{ animationFillMode: 'forwards' }}>
                <div className="flex items-center gap-2 mb-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="text-sm font-medium">Output</span>
                  {selectedTask.status === "running" && (
                    <span className="text-xs text-blue-400 flex items-center gap-1">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75" />
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500" />
                      </span>
                      Live
                    </span>
                  )}
                </div>
                <pre
                  ref={outputRef}
                  className="bg-slate-950 p-4 rounded-xl text-sm text-emerald-400 overflow-auto max-h-80 whitespace-pre-wrap font-mono border border-slate-800 transition-all duration-300"
                >
                  {selectedTask.status === "running"
                    ? getLiveOutputForTask(selectedTask.task_id) || "Waiting for output..."
                    : selectedTask.stdout || "(No output)"}
                </pre>
              </div>

              {selectedTask.stderr && (
                <div className="mb-4 opacity-0 animate-fade-in-up stagger-3" style={{ animationFillMode: 'forwards' }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-amber-500" />
                    <span className="text-sm font-medium">Stderr</span>
                  </div>
                  <pre className="bg-slate-950 p-4 rounded-xl text-sm text-amber-400 overflow-auto max-h-40 whitespace-pre-wrap font-mono border border-slate-800">
                    {selectedTask.stderr}
                  </pre>
                </div>
              )}

              {selectedTask.error && (
                <div className="opacity-0 animate-fade-in-up stagger-4" style={{ animationFillMode: 'forwards' }}>
                  <div className="flex items-center gap-2 mb-2">
                    <span className="w-2 h-2 rounded-full bg-rose-500" />
                    <span className="text-sm font-medium">Error</span>
                  </div>
                  <pre className="bg-rose-950/30 p-4 rounded-xl text-sm text-rose-400 border border-rose-900/50">
                    {selectedTask.error}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Model Detail Modal */}
      {selectedModel && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 animate-backdrop-in"
          onClick={() => setSelectedModel(null)}
        >
          <div
            className="bg-slate-900 rounded-2xl max-w-2xl w-full overflow-hidden border border-slate-700/50 shadow-2xl animate-modal-in"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/50">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center text-lg">
                  🧠
                </div>
                <div>
                  <h3 className="font-semibold text-lg">{selectedModel.name}</h3>
                  <p className="text-xs text-slate-500 font-mono">{selectedModel.model_id}</p>
                </div>
              </div>
              <button
                className="w-8 h-8 rounded-lg bg-slate-700/50 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-all duration-200 hover:rotate-90"
                onClick={() => setSelectedModel(null)}
              >
                ✕
              </button>
            </div>

            <div className="p-6">
              <div className="grid grid-cols-2 gap-4 mb-6">
                {[
                  { label: "Status", value: selectedModel.status },
                  { label: "Type", value: selectedModel.model_type || "general" },
                  { label: "Framework", value: selectedModel.framework },
                  { label: "Size", value: formatFileSize(selectedModel.file_size) },
                  { label: "Created", value: formatDate(selectedModel.created_at) },
                  { label: "Updated", value: formatDate(selectedModel.updated_at) },
                ].map((item) => (
                  <div key={item.label} className="bg-slate-800/50 rounded-xl p-4">
                    <p className="text-xs text-slate-400 mb-1">{item.label}</p>
                    <p className="font-medium">{item.value}</p>
                  </div>
                ))}
              </div>

              {selectedModel.description && (
                <div className="mb-6">
                  <p className="text-xs text-slate-400 mb-2">Description</p>
                  <p className="text-sm text-slate-300">{selectedModel.description}</p>
                </div>
              )}

              <div className="flex gap-3">
                <button
                  className="flex-1 px-4 py-3 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 rounded-xl font-medium transition-all duration-300"
                  onClick={() => {
                    setInferenceModel(selectedModel);
                    setShowInferenceModal(true);
                    setSelectedModel(null);
                  }}
                >
                  Run Inference
                </button>
                <button
                  className="px-4 py-3 bg-slate-700/50 hover:bg-slate-600/50 rounded-xl font-medium transition-colors"
                  onClick={() => downloadModel(selectedModel.model_id)}
                >
                  Download
                </button>
                <button
                  className="px-4 py-3 bg-rose-500/20 text-rose-400 hover:bg-rose-500/30 rounded-xl font-medium transition-colors"
                  onClick={() => {
                    deleteModel(selectedModel.model_id);
                  }}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model Register Modal */}
      {showModelRegister && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 animate-backdrop-in"
          onClick={() => setShowModelRegister(false)}
        >
          <div
            className="bg-slate-900 rounded-2xl max-w-lg w-full overflow-hidden border border-slate-700/50 shadow-2xl animate-modal-in"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/50">
              <h3 className="font-semibold text-lg">Register New Model</h3>
              <button
                className="w-8 h-8 rounded-lg bg-slate-700/50 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-all duration-200 hover:rotate-90"
                onClick={() => setShowModelRegister(false)}
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Model Name</label>
                <input
                  type="text"
                  className="w-full bg-slate-800/50 text-white p-3 rounded-xl border border-slate-700/50 focus:border-purple-500/50 focus:outline-none transition-all"
                  placeholder="e.g., mnist-classifier-v1"
                  value={modelName}
                  onChange={(e) => setModelName(e.target.value)}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Model Type</label>
                <select
                  className="w-full bg-slate-800/50 text-white p-3 rounded-xl border border-slate-700/50 focus:border-purple-500/50 focus:outline-none transition-all"
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                >
                  <option value="pytorch">PyTorch</option>
                  <option value="classifier">Classifier</option>
                  <option value="detector">Detector</option>
                  <option value="generator">Generator</option>
                  <option value="transformer">Transformer</option>
                  <option value="other">Other</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Model File</label>
                <div className="border-2 border-dashed border-slate-600/50 hover:border-purple-500/50 rounded-xl p-6 text-center transition-all cursor-pointer">
                  <input
                    type="file"
                    accept=".pt,.pth,.onnx,.pkl"
                    className="hidden"
                    id="model-file-upload"
                    onChange={(e) => setModelFile(e.target.files?.[0] || null)}
                  />
                  <label htmlFor="model-file-upload" className="cursor-pointer block">
                    {modelFile ? (
                      <div>
                        <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-purple-500/20 flex items-center justify-center">
                          <span className="text-purple-400 text-xl">🧠</span>
                        </div>
                        <p className="text-purple-400 font-medium">{modelFile.name}</p>
                        <p className="text-xs text-slate-500 mt-1">{formatFileSize(modelFile.size)}</p>
                      </div>
                    ) : (
                      <div>
                        <div className="w-12 h-12 mx-auto mb-2 rounded-xl bg-slate-700/50 flex items-center justify-center">
                          <span className="text-slate-400 text-xl">📦</span>
                        </div>
                        <p className="text-slate-400">Click to select model file</p>
                        <p className="text-xs text-slate-600 mt-1">.pt, .pth, .onnx, .pkl</p>
                      </div>
                    )}
                  </label>
                </div>
              </div>

              <button
                className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-slate-600 disabled:to-slate-700 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-purple-500/25 disabled:shadow-none"
                onClick={registerModel}
                disabled={isSubmitting || !modelFile || !modelName.trim()}
              >
                {isSubmitting ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Registering...
                  </span>
                ) : "Register Model"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Inference Modal */}
      {showInferenceModal && inferenceModel && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center p-4 z-50 animate-backdrop-in"
          onClick={() => setShowInferenceModal(false)}
        >
          <div
            className="bg-slate-900 rounded-2xl max-w-lg w-full overflow-hidden border border-slate-700/50 shadow-2xl animate-modal-in"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center px-6 py-4 border-b border-slate-700/50 bg-slate-800/50">
              <div>
                <h3 className="font-semibold text-lg">Run Inference</h3>
                <p className="text-xs text-slate-500">{inferenceModel.name}</p>
              </div>
              <button
                className="w-8 h-8 rounded-lg bg-slate-700/50 hover:bg-slate-600 flex items-center justify-center text-slate-400 hover:text-white transition-all duration-200 hover:rotate-90"
                onClick={() => setShowInferenceModal(false)}
              >
                ✕
              </button>
            </div>

            <div className="p-6 space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-400 mb-2">Input Data (JSON)</label>
                <textarea
                  className="w-full h-40 bg-slate-800/50 text-emerald-400 font-mono text-sm p-4 rounded-xl border border-slate-700/50 focus:border-purple-500/50 focus:outline-none resize-none transition-all"
                  placeholder='{"image": "base64...", "batch_size": 1}'
                  value={inferenceInput}
                  onChange={(e) => setInferenceInput(e.target.value)}
                />
              </div>

              <button
                className="w-full py-3 bg-gradient-to-r from-purple-500 to-pink-600 hover:from-purple-600 hover:to-pink-700 disabled:from-slate-600 disabled:to-slate-700 rounded-xl font-semibold transition-all duration-300 shadow-lg shadow-purple-500/25 disabled:shadow-none"
                onClick={runInference}
                disabled={isSubmitting}
              >
                {isSubmitting ? (
                  <span className="flex items-center justify-center gap-2">
                    <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    Starting...
                  </span>
                ) : "Run Inference"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
