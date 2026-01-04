"use client";

import { useState, useEffect, useRef, useCallback } from "react";

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
  status: string;
  gpu_id: number | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  stdout: string | null;
  stderr: string | null;
  return_code: number | null;
  error: string | null;
}

export default function Dashboard() {
  const [connected, setConnected] = useState(false);
  const [gpus, setGpus] = useState<GPU[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [liveOutput, setLiveOutput] = useState<{ [key: string]: string[] }>({});
  const [code, setCode] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [entryPoint, setEntryPoint] = useState("main.py");
  const [activeTab, setActiveTab] = useState<"code" | "file">("code");
  const wsRef = useRef<WebSocket | null>(null);
  const outputRef = useRef<HTMLPreElement>(null);

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

  const connectWebSocket = useCallback(() => {
    const ws = new WebSocket(getWsUrl());

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onclose = () => {
      setConnected(false);
      // 재연결 시도
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
          // 선택된 작업 업데이트
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
    return () => {
      wsRef.current?.close();
    };
  }, [connectWebSocket]);

  // 자동 스크롤
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
      alert("작업 제출 실패");
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
      alert("파일 업로드 실패");
    }
    setIsSubmitting(false);
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "completed": return "bg-green-500";
      case "running": return "bg-blue-500 animate-pulse";
      case "queued": return "bg-yellow-500";
      case "pending": return "bg-gray-400";
      case "failed": return "bg-red-500";
      default: return "bg-gray-400";
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case "completed": return "완료";
      case "running": return "실행중";
      case "queued": return "대기중";
      case "pending": return "준비중";
      case "failed": return "실패";
      default: return status;
    }
  };

  const formatTime = (dateStr: string | null) => {
    if (!dateStr) return "-";
    return new Date(dateStr).toLocaleTimeString("ko-KR");
  };

  const getLiveOutputForTask = (taskId: string) => {
    return liveOutput[taskId]?.join("") || "";
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold">VrunGPU Dashboard</h1>
          <p className="text-gray-400">원격 GPU 실행 서버 모니터링</p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`inline-block w-3 h-3 rounded-full ${
              connected ? "bg-green-500 animate-pulse" : "bg-red-500"
            }`}
          />
          <span className={connected ? "text-green-400" : "text-red-400"}>
            {connected ? "실시간 연결됨" : "연결 끊김"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* GPU Status */}
        <div className="lg:col-span-2">
          <h2 className="text-xl font-semibold mb-4">GPU 상태</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {gpus.map((gpu) => (
              <div
                key={gpu.index}
                className={`bg-gray-800 rounded-lg p-4 border-2 transition-all ${
                  gpu.task_id ? "border-blue-500 shadow-lg shadow-blue-500/20" : "border-gray-700"
                }`}
              >
                <div className="flex justify-between items-start mb-3">
                  <div>
                    <h3 className="font-semibold">GPU {gpu.index}</h3>
                    <p className="text-sm text-gray-400">{gpu.name}</p>
                  </div>
                  <span
                    className={`px-2 py-1 rounded text-xs ${
                      gpu.task_id
                        ? "bg-blue-500/20 text-blue-400"
                        : "bg-green-500/20 text-green-400"
                    }`}
                  >
                    {gpu.task_id ? "사용중" : "대기"}
                  </span>
                </div>

                <div className="mb-2">
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>메모리</span>
                    <span>{gpu.memory_used_mb} / {gpu.memory_total_mb} MB</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-blue-500 transition-all duration-500"
                      style={{ width: `${(gpu.memory_used_mb / gpu.memory_total_mb) * 100}%` }}
                    />
                  </div>
                </div>

                <div>
                  <div className="flex justify-between text-xs text-gray-400 mb-1">
                    <span>사용률</span>
                    <span>{gpu.utilization_percent}%</span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-500 ${
                        gpu.utilization_percent > 80
                          ? "bg-red-500"
                          : gpu.utilization_percent > 50
                          ? "bg-yellow-500"
                          : "bg-green-500"
                      }`}
                      style={{ width: `${gpu.utilization_percent}%` }}
                    />
                  </div>
                </div>

                {gpu.task_id && (
                  <p className="text-xs text-blue-400 mt-2 truncate">
                    Task: {gpu.task_id.slice(0, 8)}...
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Submit Task */}
        <div>
          <h2 className="text-xl font-semibold mb-4">작업 제출</h2>
          <div className="bg-gray-800 rounded-lg p-4">
            {/* Tabs */}
            <div className="flex mb-4 border-b border-gray-700">
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === "code"
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-400"
                }`}
                onClick={() => setActiveTab("code")}
              >
                코드 입력
              </button>
              <button
                className={`px-4 py-2 text-sm font-medium ${
                  activeTab === "file"
                    ? "text-blue-400 border-b-2 border-blue-400"
                    : "text-gray-400"
                }`}
                onClick={() => setActiveTab("file")}
              >
                파일 업로드
              </button>
            </div>

            {activeTab === "code" ? (
              <>
                <textarea
                  className="w-full h-40 bg-gray-900 text-green-400 font-mono text-sm p-3 rounded border border-gray-700 focus:border-blue-500 focus:outline-none resize-none"
                  placeholder="Python 코드를 입력하세요..."
                  value={code}
                  onChange={(e) => setCode(e.target.value)}
                />
                <button
                  className="w-full mt-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-2 px-4 rounded font-semibold transition-colors"
                  onClick={submitCode}
                  disabled={isSubmitting || !code.trim()}
                >
                  {isSubmitting ? "제출 중..." : "실행"}
                </button>
              </>
            ) : (
              <>
                <div className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center mb-3">
                  <input
                    type="file"
                    accept=".py,.zip"
                    className="hidden"
                    id="file-upload"
                    onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                  />
                  <label
                    htmlFor="file-upload"
                    className="cursor-pointer block"
                  >
                    {uploadFile ? (
                      <div>
                        <p className="text-blue-400 font-medium">{uploadFile.name}</p>
                        <p className="text-xs text-gray-500">
                          {(uploadFile.size / 1024).toFixed(1)} KB
                        </p>
                      </div>
                    ) : (
                      <div>
                        <p className="text-gray-400">클릭하여 파일 선택</p>
                        <p className="text-xs text-gray-500 mt-1">
                          .py 또는 .zip (데이터셋 포함 가능)
                        </p>
                      </div>
                    )}
                  </label>
                </div>
                <input
                  type="text"
                  className="w-full bg-gray-900 text-white text-sm p-2 rounded border border-gray-700 focus:border-blue-500 focus:outline-none mb-3"
                  placeholder="Entry point (예: train.py)"
                  value={entryPoint}
                  onChange={(e) => setEntryPoint(e.target.value)}
                />
                <button
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 py-2 px-4 rounded font-semibold transition-colors"
                  onClick={submitFile}
                  disabled={isSubmitting || !uploadFile}
                >
                  {isSubmitting ? "업로드 중..." : "업로드 및 실행"}
                </button>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Task List */}
      <div className="mt-8">
        <h2 className="text-xl font-semibold mb-4">작업 목록</h2>
        <div className="bg-gray-800 rounded-lg overflow-hidden">
          <table className="w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-sm">Task ID</th>
                <th className="px-4 py-3 text-left text-sm">상태</th>
                <th className="px-4 py-3 text-left text-sm">GPU</th>
                <th className="px-4 py-3 text-left text-sm">시작</th>
                <th className="px-4 py-3 text-left text-sm">완료</th>
                <th className="px-4 py-3 text-left text-sm">결과</th>
              </tr>
            </thead>
            <tbody>
              {tasks.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                    작업이 없습니다
                  </td>
                </tr>
              ) : (
                tasks.map((task) => (
                  <tr
                    key={task.task_id}
                    className="border-t border-gray-700 hover:bg-gray-700/50 cursor-pointer"
                    onClick={() => setSelectedTask(task)}
                  >
                    <td className="px-4 py-3 font-mono text-sm">
                      {task.task_id.slice(0, 8)}...
                    </td>
                    <td className="px-4 py-3">
                      <span className="inline-flex items-center gap-2 px-2 py-1 rounded text-xs">
                        <span className={`w-2 h-2 rounded-full ${getStatusColor(task.status)}`} />
                        {getStatusText(task.status)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm">
                      {task.gpu_id !== null ? `GPU ${task.gpu_id}` : "-"}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatTime(task.started_at)}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {formatTime(task.completed_at)}
                    </td>
                    <td className="px-4 py-3 text-sm">
                      {task.return_code !== null ? (
                        <span className={task.return_code === 0 ? "text-green-400" : "text-red-400"}>
                          {task.return_code === 0 ? "성공" : `오류 (${task.return_code})`}
                        </span>
                      ) : task.status === "running" ? (
                        <span className="text-blue-400">실행중...</span>
                      ) : (
                        "-"
                      )}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Task Detail Modal */}
      {selectedTask && (
        <div
          className="fixed inset-0 bg-black/70 flex items-center justify-center p-4 z-50"
          onClick={() => setSelectedTask(null)}
        >
          <div
            className="bg-gray-800 rounded-lg max-w-4xl w-full max-h-[85vh] overflow-hidden flex flex-col"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex justify-between items-center p-4 border-b border-gray-700">
              <div className="flex items-center gap-3">
                <h3 className="font-semibold">작업 상세</h3>
                <span className="inline-flex items-center gap-2 px-2 py-1 rounded text-xs bg-gray-700">
                  <span className={`w-2 h-2 rounded-full ${getStatusColor(selectedTask.status)}`} />
                  {getStatusText(selectedTask.status)}
                </span>
              </div>
              <button
                className="text-gray-400 hover:text-white text-xl"
                onClick={() => setSelectedTask(null)}
              >
                ✕
              </button>
            </div>
            <div className="p-4 overflow-y-auto flex-1">
              <div className="grid grid-cols-4 gap-4 mb-4 text-sm">
                <div>
                  <span className="text-gray-400">Task ID</span>
                  <p className="font-mono text-xs">{selectedTask.task_id}</p>
                </div>
                <div>
                  <span className="text-gray-400">GPU</span>
                  <p>{selectedTask.gpu_id !== null ? `GPU ${selectedTask.gpu_id}` : "-"}</p>
                </div>
                <div>
                  <span className="text-gray-400">시작</span>
                  <p>{formatTime(selectedTask.started_at)}</p>
                </div>
                <div>
                  <span className="text-gray-400">Return Code</span>
                  <p>{selectedTask.return_code ?? "-"}</p>
                </div>
              </div>

              <div className="mb-4">
                <span className="text-gray-400 text-sm">Output (실시간)</span>
                <pre
                  ref={outputRef}
                  className="bg-gray-900 p-3 rounded mt-1 text-sm text-green-400 overflow-auto max-h-80 whitespace-pre-wrap font-mono"
                >
                  {selectedTask.status === "running"
                    ? getLiveOutputForTask(selectedTask.task_id) || "출력 대기중..."
                    : selectedTask.stdout || "(출력 없음)"}
                </pre>
              </div>

              {(selectedTask.stderr || (selectedTask.status === "running" && liveOutput[selectedTask.task_id])) && (
                <div className="mb-4">
                  <span className="text-gray-400 text-sm">Stderr</span>
                  <pre className="bg-gray-900 p-3 rounded mt-1 text-sm text-yellow-400 overflow-auto max-h-40 whitespace-pre-wrap font-mono">
                    {selectedTask.stderr || ""}
                  </pre>
                </div>
              )}

              {selectedTask.error && (
                <div>
                  <span className="text-gray-400 text-sm">Error</span>
                  <pre className="bg-red-900/30 p-3 rounded mt-1 text-sm text-red-400">
                    {selectedTask.error}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
