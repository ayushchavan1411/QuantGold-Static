"use client";

import React, { useState, useEffect } from "react";
import dynamic from "next/dynamic";

const ResponsiveContainer = dynamic(
  () => import("recharts").then((mod) => mod.ResponsiveContainer),
  { ssr: false }
);
const LineChart = dynamic(() => import("recharts").then((mod) => mod.LineChart), { ssr: false });
const Line = dynamic(() => import("recharts").then((mod) => mod.Line), { ssr: false });
const XAxis = dynamic(() => import("recharts").then((mod) => mod.XAxis), { ssr: false });
const YAxis = dynamic(() => import("recharts").then((mod) => mod.YAxis), { ssr: false });
const CartesianGrid = dynamic(() => import("recharts").then((mod) => mod.CartesianGrid), { ssr: false });
const Tooltip = dynamic(() => import("recharts").then((mod) => mod.Tooltip), { ssr: false });

export default function TradingDashboard() {
  const [data, setData] = useState<any>(null);
  const [status, setStatus] = useState<string>("idle");
  const [error, setError] = useState<string>("");
  const [expandedSession, setExpandedSession] = useState<string | null>(null); // Toggle state

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (status === "running") {
      interval = setInterval(fetchData, 1000);
    }
    return () => clearInterval(interval);
  }, [status]);

  const fetchData = async () => {
    try {
      const res = await fetch("http://localhost:8000/data");
      if (!res.ok) throw new Error("Network response was not ok");
      const result = await res.json();
      setData(result);
      setStatus(result.status);
      setError("");
    } catch (err) {
      setError("Backend unreachable — ensure FastAPI is running on port 8000.");
      setStatus("error");
    }
  };

  const runBacktest = async () => {
    try {
      setStatus("running");
      await fetch("http://localhost:8000/trigger", { method: "POST" });
    } catch (err) {
      setError("Failed to trigger backtest.");
      setStatus("error");
    }
  };

  const formatMoney = (val: any) => {
    const num = parseFloat(val);
    if (isNaN(num)) return "$0.00";
    return num >= 0 ? `$${num.toFixed(2)}` : `-$${Math.abs(num).toFixed(2)}`;
  };

  const calculateRowStats = (statsObj: any) => {
    const W = statsObj.W || 0;
    const L = statsObj.L || 0;
    const total = W + L;
    const grossWin = statsObj.gross_win || 0;
    const grossLoss = statsObj.gross_loss || 0;
    
    const wr = total > 0 ? ((W / total) * 100).toFixed(1) : "0.0";
    const pf = grossLoss > 0 ? (grossWin / grossLoss).toFixed(2) : (grossWin > 0 ? "∞" : "0.00");
    const avgWin = W > 0 ? formatMoney(grossWin / W) : "$0.00";
    const avgLoss = L > 0 ? formatMoney(grossLoss / L) : "$0.00";

    return { W, L, wr, pf, avgWin, avgLoss };
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-gray-300 font-mono p-6 selection:bg-yellow-500/30">
      
      {/* Error Banner */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 mb-6 rounded-md flex items-center">
          <span className="mr-2">⚠️</span> {error}
        </div>
      )}

      {/* Header Panel */}
      <div className="border border-gray-800 bg-[#111] p-4 flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-1">Strategy</h2>
          <div className="text-yellow-500 font-bold">V26 SMC (1D FIB 1.0 & 0.382 + 5M EMA)</div>
        </div>
        <div>
          <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-1">Asset Configuration</h2>
          <div className="text-white font-bold">XAU/USD (50x LEV | $200 MARGIN)</div>
        </div>
        <button 
          onClick={runBacktest}
          disabled={status === "running"}
          className={`px-6 py-2 border font-bold uppercase transition-all duration-200 ${
            status === "running" 
            ? "border-gray-600 text-gray-600 cursor-not-allowed" 
            : "border-yellow-500 text-yellow-500 hover:bg-yellow-500 hover:text-black"
          }`}
        >
          {status === "running" ? "Crunching MT5 Data..." : "Run Backtest"}
        </button>
      </div>

      {/* Expanded Top Stats Row (8 Panels) */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Account Equity</div>
          <div className="text-2xl md:text-3xl font-bold text-white mb-1">
            {data?.stats?.paper_equity || "$500.00"}
          </div>
          <div className={`text-[10px] md:text-xs ${data?.stats?.net_profit?.includes("-") ? "text-red-500" : "text-green-500"}`}>
            net profit: {data?.stats?.net_profit || "$0.00"}
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Win Rate</div>
          <div className="text-2xl md:text-3xl font-bold text-white mb-1">
            {data?.stats?.win_rate || "0.0%"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            {data?.stats?.wl_ratio || "0W / 0L"}
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Total Trades</div>
          <div className="text-2xl md:text-3xl font-bold text-white mb-1">
            {data?.stats?.total_trades || "0"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            expectancy: {data?.stats?.expectancy || "$0.00"}
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Profit Factor</div>
          <div className="text-2xl md:text-3xl font-bold text-green-500 mb-1">
            {data?.stats?.profit_factor || "0.00"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            gross win / loss
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Max Drawdown</div>
          <div className="text-2xl md:text-3xl font-bold text-red-500 mb-1">
            {data?.stats?.max_drawdown || "0.00%"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            peak to trough
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Sharpe Ratio</div>
          <div className="text-2xl md:text-3xl font-bold text-yellow-500 mb-1">
            {data?.stats?.sharpe || "0.00"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            risk-adjusted return
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Avg Win</div>
          <div className="text-2xl md:text-3xl font-bold text-green-500 mb-1">
            {data?.stats?.avg_win || "$0.00"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            per winning trade
          </div>
        </div>
        <div className="border border-gray-800 bg-[#111] p-4">
          <div className="text-[10px] md:text-xs text-gray-500 uppercase tracking-widest mb-2">Avg Loss</div>
          <div className="text-2xl md:text-3xl font-bold text-red-500 mb-1">
            {data?.stats?.avg_loss || "$0.00"}
          </div>
          <div className="text-[10px] md:text-xs text-gray-500">
            per losing trade
          </div>
        </div>
      </div>

      {/* Chart & Ledger Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="lg:col-span-2 border border-gray-800 bg-[#111] p-4">
          <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-4">Equity Curve (USD)</h2>
          <div className="h-[400px] w-full">
            {data?.equity_curve?.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={data.equity_curve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#222" vertical={false} />
                  <XAxis dataKey="date" stroke="#555" tick={{ fill: '#777', fontSize: 12 }} minTickGap={30} />
                  <YAxis stroke="#555" tick={{ fill: '#777', fontSize: 12 }} domain={['auto', 'auto']} tickFormatter={(val) => `$${val}`} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#000', border: '1px solid #333', color: '#fff' }}
                    itemStyle={{ color: '#eab308' }}
                  />
                  <Line type="stepAfter" dataKey="equity" stroke="#eab308" strokeWidth={2} dot={false} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-600">Awaiting data...</div>
            )}
          </div>
        </div>

        <div className="border border-gray-800 bg-[#111] p-4 overflow-hidden flex flex-col h-auto lg:h-[465px]">
          <h2 className="text-xs text-gray-500 uppercase tracking-widest mb-4">Trade Ledger</h2>
          <div className="flex-1 overflow-y-auto pr-2 space-y-4 custom-scrollbar">
            {data?.logs?.length > 0 ? (
              data.logs.map((log: any, i: number) => (
                <div key={i} className="border-b border-gray-800 pb-3">
                  <div className="flex justify-between items-start mb-1">
                    <div className="flex items-center gap-2">
                      <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold text-black ${log.action.includes('LONG') ? 'bg-green-500' : log.action.includes('SHORT') ? 'bg-red-500' : 'bg-yellow-500'}`}>
                        {log.type}
                      </span>
                      <span className="text-sm text-white font-bold">{log.action}</span>
                    </div>
                    {log.profit && (
                      <span className={`text-sm font-bold ${log.profit.includes("-") ? "text-red-500" : "text-green-500"}`}>
                        {log.profit}
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-400">{log.detail}</div>
                </div>
              ))
            ) : (
              <div className="h-full flex flex-col items-center justify-center text-gray-600 text-sm">
                <span>◎</span>
                <span className="mt-2">Data unavailable</span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Detailed Breakdown Tables */}
      {data?.stats?.matrix_breakdown && (
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          
          <div className="border border-gray-800 bg-[#111] p-4">
            <h2 className="text-xs text-yellow-500 uppercase tracking-widest mb-4 border-b border-gray-800 pb-2">Breakdown By Setup (Fibonacci)</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-500 uppercase bg-[#0a0a0a]">
                  <tr>
                    <th className="px-4 py-2 font-normal">Setup</th>
                    <th className="px-4 py-2 font-normal text-center">W</th>
                    <th className="px-4 py-2 font-normal text-center">L</th>
                    <th className="px-4 py-2 font-normal text-right">Win %</th>
                    <th className="px-4 py-2 font-normal text-right">PF</th>
                    <th className="px-4 py-2 font-normal text-right">Avg Win</th>
                    <th className="px-4 py-2 font-normal text-right">Avg Loss</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.stats.setup_breakdown).map((entry: any, idx: number) => {
                    const setup = entry[0];
                    const stats = entry[1];
                    const row = calculateRowStats(stats);
                    return (
                      <tr key={idx} className="border-b border-gray-800/50 hover:bg-[#1a1a1a]">
                        <td className="px-4 py-3 font-medium text-gray-300">{setup}</td>
                        <td className="px-4 py-3 text-center text-green-500">{row.W}</td>
                        <td className="px-4 py-3 text-center text-red-500">{row.L}</td>
                        <td className="px-4 py-3 text-right text-gray-300">{row.wr}%</td>
                        <td className="px-4 py-3 text-right text-yellow-500">{row.pf}</td>
                        <td className="px-4 py-3 text-right text-green-500">{row.avgWin}</td>
                        <td className="px-4 py-3 text-right text-red-500">{row.avgLoss}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          <div className="border border-gray-800 bg-[#111] p-4">
            <h2 className="text-xs text-yellow-500 uppercase tracking-widest mb-4 border-b border-gray-800 pb-2">Deep Matrix: Session vs Setup</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-gray-500 uppercase bg-[#0a0a0a]">
                  <tr>
                    <th className="px-4 py-2 font-normal">Session</th>
                    <th className="px-4 py-2 font-normal text-center">W</th>
                    <th className="px-4 py-2 font-normal text-center">L</th>
                    <th className="px-4 py-2 font-normal text-right">Win %</th>
                    <th className="px-4 py-2 font-normal text-right">PF</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(data.stats.session_breakdown).map((entry: any, idx: number) => {
                    const session = entry[0];
                    const sessionStats = entry[1];
                    const row = calculateRowStats(sessionStats);
                    const isExpanded = expandedSession === session;
                    const setupsInSession = data.stats.matrix_breakdown[session];

                    return (
                      <React.Fragment key={idx}>
                        <tr 
                          onClick={() => setExpandedSession(isExpanded ? null : session)}
                          className="border-b border-gray-800/50 hover:bg-[#1a1a1a] cursor-pointer"
                        >
                          <td className="px-4 py-3 font-medium text-gray-300 flex items-center gap-2">
                             <span className="text-[10px]">{isExpanded ? "▼" : "▶"}</span> {session}
                          </td>
                          <td className="px-4 py-3 text-center text-green-500">{row.W}</td>
                          <td className="px-4 py-3 text-center text-red-500">{row.L}</td>
                          <td className="px-4 py-3 text-right text-gray-300">{row.wr}%</td>
                          <td className="px-4 py-3 text-right text-yellow-500">{row.pf}</td>
                        </tr>
                        
                        {/* Nested Dropdown Rows */}
                        {isExpanded && Object.entries(setupsInSession).map((subEntry: any, subIdx: number) => {
                           const setupName = subEntry[0];
                           const setupStats = subEntry[1];
                           const subRow = calculateRowStats(setupStats);
                           if (subRow.W === 0 && subRow.L === 0) return null; // Hide empty rows
                           return (
                             <tr key={`${idx}-${subIdx}`} className="bg-[#0f0f0f] border-b border-gray-800/30 text-xs">
                               <td className="px-8 py-2 text-gray-500">↳ {setupName}</td>
                               <td className="px-4 py-2 text-center text-green-500/70">{subRow.W}</td>
                               <td className="px-4 py-2 text-center text-red-500/70">{subRow.L}</td>
                               <td className="px-4 py-2 text-right text-gray-400">{subRow.wr}%</td>
                               <td className="px-4 py-2 text-right text-yellow-500/70">{subRow.pf}</td>
                             </tr>
                           )
                        })}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

        </div>
      )}

      <style dangerouslySetInnerHTML={{__html: `
        .custom-scrollbar::-webkit-scrollbar { width: 4px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: #111; }
        .custom-scrollbar::-webkit-scrollbar-thumb { background: #333; border-radius: 4px; }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #555; }
      `}} />
    </div>
  );
}