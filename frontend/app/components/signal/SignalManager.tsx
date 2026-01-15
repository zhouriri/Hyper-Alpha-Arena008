import { useEffect, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'react-hot-toast'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Switch } from '@/components/ui/switch'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Plus, Trash2, Edit, Activity, Eye, Sparkles } from 'lucide-react'
import SignalPreviewChart from './SignalPreviewChart'
import AiSignalChatModal from './AiSignalChatModal'
import MarketRegimeConfig from './MarketRegimeConfig'
import PacmanLoader from '../ui/pacman-loader'

// Types
interface SignalDefinition {
  id: number
  signal_name: string
  description: string | null
  trigger_condition: TriggerCondition
  enabled: boolean
  created_at: string
  updated_at: string
}

interface TriggerCondition {
  metric?: string
  operator?: string
  threshold?: number
  time_window?: string
  logic?: string
  conditions?: TriggerCondition[]
}

interface SignalPool {
  id: number
  pool_name: string
  signal_ids: number[]
  symbols: string[]
  enabled: boolean
  logic: 'OR' | 'AND'
  created_at: string
}

interface MarketRegimeData {
  regime: string
  direction: string
  confidence: number
  details?: Record<string, unknown>
}

interface SignalTriggerLog {
  id: number
  signal_id: number | null
  pool_id: number | null
  symbol: string
  trigger_value: Record<string, unknown> | null
  triggered_at: string
  market_regime: MarketRegimeData | null
}

// API functions
const API_BASE = '/api/signals'

async function fetchSignals(): Promise<{ signals: SignalDefinition[]; pools: SignalPool[] }> {
  const res = await fetch(API_BASE)
  if (!res.ok) throw new Error('Failed to fetch signals')
  return res.json()
}

async function createSignal(data: Partial<SignalDefinition>): Promise<SignalDefinition> {
  const res = await fetch(`${API_BASE}/definitions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create signal')
  return res.json()
}

async function updateSignal(id: number, data: Partial<SignalDefinition>): Promise<SignalDefinition> {
  const res = await fetch(`${API_BASE}/definitions/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to update signal')
  return res.json()
}

async function deleteSignal(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/definitions/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete signal')
}

async function createPool(data: Partial<SignalPool>): Promise<SignalPool> {
  const res = await fetch(`${API_BASE}/pools`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to create pool')
  return res.json()
}

async function updatePool(id: number, data: Partial<SignalPool>): Promise<SignalPool> {
  const res = await fetch(`${API_BASE}/pools/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
  if (!res.ok) throw new Error('Failed to update pool')
  return res.json()
}

async function deletePool(id: number): Promise<void> {
  const res = await fetch(`${API_BASE}/pools/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error('Failed to delete pool')
}

// Create signal pool from AI-generated config
async function createPoolFromConfig(config: {
  name: string
  symbol: string
  description?: string
  logic: string
  signals: Array<{ metric: string; operator: string; threshold: number; time_window?: string }>
}): Promise<{ success: boolean; pool: SignalPool; signals: SignalDefinition[] }> {
  const res = await fetch(`${API_BASE}/create-pool-from-config`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  })
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: 'Failed to create pool' }))
    throw new Error(error.detail || 'Failed to create pool')
  }
  return res.json()
}

async function fetchTriggerLogs(poolId?: number, limit = 50): Promise<SignalTriggerLog[]> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (poolId) params.set('pool_id', String(poolId))
  const res = await fetch(`${API_BASE}/logs?${params}`)
  if (!res.ok) throw new Error('Failed to fetch logs')
  const data = await res.json()
  return data.logs
}

async function fetchPoolBacktest(poolId: number, symbol: string): Promise<any> {
  const params = new URLSearchParams({ symbol })
  const res = await fetch(`${API_BASE}/pool-backtest/${poolId}?${params}`)
  if (!res.ok) throw new Error('Failed to fetch pool backtest')
  return res.json()
}

// Market Regime batch query
interface MarketRegimeResult {
  symbol: string
  regime: string
  direction: string
  confidence: number
  reason: string
}

async function fetchBatchMarketRegime(
  symbols: string[],
  timeframe: string,
  timestamps: number[]
): Promise<Map<number, MarketRegimeResult>> {
  const results = new Map<number, MarketRegimeResult>()
  // Query regime for each unique timestamp
  const uniqueTimestamps = [...new Set(timestamps)]
  for (const ts of uniqueTimestamps) {
    try {
      const res = await fetch('/api/market-regime/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbols,
          timeframe,
          timestamp_ms: ts,
        }),
      })
      if (res.ok) {
        const data = await res.json()
        if (data.results && data.results.length > 0) {
          results.set(ts, data.results[0])
        }
      }
    } catch (e) {
      console.error(`Failed to fetch regime for timestamp ${ts}:`, e)
    }
  }
  return results
}

interface MetricAnalysis {
  status: string
  symbol: string
  metric: string
  period: string
  sample_count: number
  time_range_hours: number
  warning?: string
  statistics?: {
    mean: number
    std: number
    min: number
    max: number
    abs_percentiles: { p75: number; p90: number; p95: number; p99: number }
  }
  suggestions?: {
    aggressive: { threshold: number; description: string }
    moderate: { threshold: number; description: string; recommended?: boolean }
    conservative: { threshold: number; description: string }
  }
  message?: string
}

async function fetchMetricAnalysis(symbol: string, metric: string, period: string): Promise<MetricAnalysis> {
  const params = new URLSearchParams({ symbol, metric, period })
  const res = await fetch(`${API_BASE}/analyze?${params}`)
  if (!res.ok) throw new Error('Failed to analyze metric')
  return res.json()
}

// Constants aligned with K-line indicators (MarketFlowIndicators.tsx)
const METRICS = [
  { value: 'oi_delta', label: 'OI Delta', desc: 'Open Interest change %. Positive=inflow, Negative=outflow' },
  { value: 'cvd', label: 'CVD', desc: 'Cumulative Volume Delta. Positive=buyers dominate, Negative=sellers dominate' },
  { value: 'funding', label: 'Funding Rate Change', desc: 'Funding rate change (aligned with K-line chart). Positive=rate increasing, Negative=rate decreasing' },
  { value: 'depth_ratio', label: 'Depth Ratio', desc: 'Bid/Ask depth ratio. >1=more bids, <1=more asks' },
  { value: 'taker_ratio', label: 'Taker Ratio', desc: 'Log taker ratio ln(buy/sell). >0=buyers, <0=sellers. Symmetric around 0' },
  { value: 'order_imbalance', label: 'Order Imbalance', desc: 'Order book imbalance (-1 to 1). Positive=buy pressure' },
  { value: 'oi', label: 'OI (Absolute)', desc: 'Absolute Open Interest value in USD' },
  { value: 'taker_volume', label: 'Taker Volume', desc: 'Composite signal: direction + ratio + volume threshold', isComposite: true },
  { value: 'price_change', label: 'Price Change', desc: 'Price change % over time window. Formula: (current-prev)/prev*100. Positive=up, Negative=down' },
  { value: 'volatility', label: 'Volatility', desc: 'Price volatility % over time window. Formula: (high-low)/low*100. Always positive, detects swings' },
]

// Direction options for taker_volume composite signal
const TAKER_DIRECTIONS = [
  { value: 'any', label: 'Any Direction', desc: 'Trigger on either buy or sell dominance' },
  { value: 'buy', label: 'Buy Dominant', desc: 'Only trigger when buyers dominate' },
  { value: 'sell', label: 'Sell Dominant', desc: 'Only trigger when sellers dominate' },
]

const OPERATORS = [
  { value: 'abs_greater_than', label: '|x| > (Absolute)', desc: 'Triggers when absolute value exceeds threshold (ignores direction)' },
  { value: 'greater_than', label: '> (Greater)', desc: 'Triggers when value is greater than threshold' },
  { value: 'less_than', label: '< (Less)', desc: 'Triggers when value is less than threshold' },
  { value: 'equals', label: '= (Equals)', desc: 'Triggers when value equals threshold' },
]

const TIME_WINDOWS = [
  { value: '1m', label: '1 min', desc: 'Very short-term, high noise' },
  { value: '3m', label: '3 min', desc: 'Short-term signals' },
  { value: '5m', label: '5 min', desc: 'Recommended for most signals' },
  { value: '15m', label: '15 min', desc: 'Medium-term, more reliable' },
  { value: '30m', label: '30 min', desc: 'Longer-term trends' },
  { value: '1h', label: '1 hour', desc: 'Major trend changes only' },
  { value: '2h', label: '2 hours', desc: 'Long-term trend confirmation' },
  { value: '4h', label: '4 hours', desc: 'Very long-term, major moves only' },
]
// Symbols are now loaded dynamically from Hyperliquid watchlist (see watchlistSymbols state)

export default function SignalManager() {
  const { t } = useTranslation()
  const [signals, setSignals] = useState<SignalDefinition[]>([])
  const [pools, setPools] = useState<SignalPool[]>([])
  const [logs, setLogs] = useState<SignalTriggerLog[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('signals')

  // Signal dialog state
  const [signalDialogOpen, setSignalDialogOpen] = useState(false)
  const [editingSignal, setEditingSignal] = useState<SignalDefinition | null>(null)
  const [signalForm, setSignalForm] = useState({
    signal_name: '',
    description: '',
    metric: 'oi_delta',
    operator: 'abs_greater_than',
    threshold: 5,
    time_window: '5m',
    enabled: true,
    // taker_volume composite fields
    direction: 'any',
    ratio_threshold: 1.5,
    volume_threshold: 50000,
  })

  // Pool dialog state
  const [poolDialogOpen, setPoolDialogOpen] = useState(false)
  const [editingPool, setEditingPool] = useState<SignalPool | null>(null)
  const [poolForm, setPoolForm] = useState({
    pool_name: '',
    signal_ids: [] as number[],
    symbols: [] as string[],
    enabled: true,
    logic: 'OR' as 'OR' | 'AND',
  })

  // Metric analysis state
  const [metricAnalysis, setMetricAnalysis] = useState<MetricAnalysis | null>(null)
  const [analysisLoading, setAnalysisLoading] = useState(false)

  // Signal preview state
  const [previewDialogOpen, setPreviewDialogOpen] = useState(false)
  const [previewSignal, setPreviewSignal] = useState<SignalDefinition | null>(null)
  const [previewSymbol, setPreviewSymbol] = useState('BTC')
  const [previewData, setPreviewData] = useState<any>(null)
  const [previewLoading, setPreviewLoading] = useState(false)
  const [chartTimeframe, setChartTimeframe] = useState('5m') // Independent chart timeframe

  // Save/delete loading states (for dialog buttons)
  const [savingSignal, setSavingSignal] = useState(false)
  const [savingPool, setSavingPool] = useState(false)

  // AI Signal Chat state
  const [aiChatOpen, setAiChatOpen] = useState(false)
  const [accounts, setAccounts] = useState<any[]>([])
  const [accountsLoading, setAccountsLoading] = useState(false)

  // Watchlist symbols for preview and analysis
  const [watchlistSymbols, setWatchlistSymbols] = useState<string[]>([])
  const [analysisSymbol, setAnalysisSymbol] = useState('BTC')

  // Pool preview state
  const [previewPool, setPreviewPool] = useState<SignalPool | null>(null)

  // Market Regime state
  const [regimeLoading, setRegimeLoading] = useState(false)

  const loadData = async () => {
    try {
      setLoading(true)
      const data = await fetchSignals()
      setSignals(data.signals)
      setPools(data.pools)
      const logsData = await fetchTriggerLogs()
      setLogs(logsData)
    } catch (err) {
      toast.error('Failed to load signal data')
    } finally {
      setLoading(false)
    }
  }

  // Silent refresh - no loading state, for use after save/delete operations
  const refreshDataSilently = async () => {
    try {
      const data = await fetchSignals()
      setSignals(data.signals)
      setPools(data.pools)
    } catch (err) {
      // Silent fail - data will refresh on next load
    }
  }

  const loadAccounts = async () => {
    try {
      setAccountsLoading(true)
      const res = await fetch('/api/account/list')
      if (res.ok) {
        const data = await res.json()
        // API returns array directly, not {accounts: [...]}
        setAccounts(Array.isArray(data) ? data : data.accounts || [])
      }
    } catch (err) {
      console.error('Failed to load accounts:', err)
    } finally {
      setAccountsLoading(false)
    }
  }

  // Silent refresh for logs only (no loading state)
  const refreshLogsSilently = async () => {
    try {
      const logsData = await fetchTriggerLogs()
      setLogs(logsData)
    } catch {
      // Silent fail - don't interrupt user
    }
  }

  // Load watchlist symbols
  const loadWatchlist = async () => {
    try {
      const res = await fetch('/api/hyperliquid/symbols/watchlist')
      if (res.ok) {
        const data = await res.json()
        const symbols = data.symbols || []
        setWatchlistSymbols(symbols)
        if (symbols.length > 0 && !symbols.includes(analysisSymbol)) {
          setAnalysisSymbol(symbols[0])
        }
      }
    } catch {
      // Silent fail
    }
  }

  // Check Market Regime for all triggers
  const checkMarketRegime = async () => {
    if (!previewData?.triggers?.length || !previewData?.symbol) return
    setRegimeLoading(true)
    try {
      // Get max timeframe from signal/pool config
      const timeframe = previewData.time_window || '5m'
      const timestamps = previewData.triggers.map((t: any) => t.timestamp)
      const regimeMap = await fetchBatchMarketRegime([previewData.symbol], timeframe, timestamps)
      // Update triggers with regime data
      const updatedTriggers = previewData.triggers.map((t: any) => ({
        ...t,
        market_regime: regimeMap.get(t.timestamp) || null,
      }))
      setPreviewData({ ...previewData, triggers: updatedTriggers })
      toast.success(`Checked regime for ${regimeMap.size} trigger points`)
    } catch (e) {
      toast.error('Failed to check market regime')
    } finally {
      setRegimeLoading(false)
    }
  }

  // Initial load
  useEffect(() => {
    loadData()
    loadAccounts()
    loadWatchlist()
  }, [])

  // Auto-refresh logs only when on logs tab (silent, no loading)
  useEffect(() => {
    if (activeTab !== 'logs') return
    const interval = setInterval(refreshLogsSilently, 15000)
    return () => clearInterval(interval)
  }, [activeTab])

  // Fetch metric analysis when dialog opens or metric/period/symbol changes
  useEffect(() => {
    if (!signalDialogOpen) {
      setMetricAnalysis(null)
      return
    }
    // Clear previous analysis immediately to avoid data mismatch during loading
    setMetricAnalysis(null)
    const loadAnalysis = async () => {
      setAnalysisLoading(true)
      try {
        const data = await fetchMetricAnalysis(analysisSymbol, signalForm.metric, signalForm.time_window)
        setMetricAnalysis(data)
      } catch {
        setMetricAnalysis(null)
      } finally {
        setAnalysisLoading(false)
      }
    }
    loadAnalysis()
  }, [signalDialogOpen, signalForm.metric, signalForm.time_window, analysisSymbol])

  const openSignalDialog = (signal?: SignalDefinition) => {
    if (signal) {
      setEditingSignal(signal)
      const cond = signal.trigger_condition
      // Map old metric names to new names (backward compatibility)
      const metricNameMap: Record<string, string> = {
        'oi_delta_percent': 'oi_delta',
        'funding_rate': 'funding',
        'taker_buy_ratio': 'taker_ratio',
      }
      const normalizedMetric = metricNameMap[cond.metric] || cond.metric || 'oi_delta'
      setSignalForm({
        signal_name: signal.signal_name,
        description: signal.description || '',
        metric: normalizedMetric,
        operator: cond.operator || 'abs_greater_than',
        threshold: cond.threshold ?? 5,
        time_window: cond.time_window || '5m',
        enabled: signal.enabled,
        // taker_volume composite fields
        direction: (cond as any).direction || 'any',
        ratio_threshold: (cond as any).ratio_threshold ?? 1.5,
        volume_threshold: (cond as any).volume_threshold ?? 50000,
      })
    } else {
      setEditingSignal(null)
      setSignalForm({
        signal_name: '',
        description: '',
        metric: 'oi_delta',
        operator: 'abs_greater_than',
        threshold: 5,
        time_window: '5m',
        enabled: true,
        direction: 'any',
        ratio_threshold: 1.5,
        volume_threshold: 50000,
      })
    }
    setSignalDialogOpen(true)
  }

  const handleSaveSignal = async () => {
    setSavingSignal(true)
    try {
      // Build trigger_condition based on metric type
      let trigger_condition: Record<string, unknown>
      if (signalForm.metric === 'taker_volume') {
        // Composite signal: direction + ratio + volume
        trigger_condition = {
          metric: signalForm.metric,
          direction: signalForm.direction,
          ratio_threshold: signalForm.ratio_threshold,
          volume_threshold: signalForm.volume_threshold,
          time_window: signalForm.time_window,
        }
      } else {
        // Standard signal: operator + threshold
        trigger_condition = {
          metric: signalForm.metric,
          operator: signalForm.operator,
          threshold: signalForm.threshold,
          time_window: signalForm.time_window,
        }
      }
      const data = {
        signal_name: signalForm.signal_name,
        description: signalForm.description,
        trigger_condition,
        enabled: signalForm.enabled,
      }
      if (editingSignal) {
        await updateSignal(editingSignal.id, data)
        toast.success('Signal updated')
      } else {
        await createSignal(data)
        toast.success('Signal created')
      }
      setSignalDialogOpen(false)
      refreshDataSilently()
    } catch (err) {
      toast.error('Failed to save signal')
    } finally {
      setSavingSignal(false)
    }
  }

  const handleDeleteSignal = async (id: number) => {
    if (!confirm('Delete this signal?')) return
    try {
      await deleteSignal(id)
      toast.success('Signal deleted')
      refreshDataSilently()
    } catch (err) {
      toast.error('Failed to delete signal')
    }
  }

  const openPreviewDialog = async (signal: SignalDefinition, symbol: string = 'BTC') => {
    // Get time_window from signal's trigger condition and set as default chart timeframe
    const signalTimeWindow = signal.trigger_condition?.time_window || '5m'
    setChartTimeframe(signalTimeWindow)
    setPreviewSignal(signal)
    setPreviewPool(null)
    setPreviewSymbol(symbol)
    setPreviewDialogOpen(true)
    setPreviewLoading(true)
    setPreviewData(null)

    try {
      // Step 1: Fetch K-lines from market API (ensures fresh data)
      // Use 500 klines to match the K-line page and provide more historical context
      const klineRes = await fetch(
        `/api/market/kline-with-indicators/${symbol}?market=hyperliquid&period=${signalTimeWindow}&count=500`
      )
      if (!klineRes.ok) throw new Error('Failed to fetch K-line data')
      const klineData = await klineRes.json()

      if (!klineData.klines || klineData.klines.length === 0) {
        throw new Error('No K-line data available')
      }

      // Get time range from K-lines (timestamps are in seconds from market API)
      const klines = klineData.klines
      const klineMinTs = Math.min(...klines.map((k: any) => k.timestamp)) * 1000
      const klineMaxTs = Math.max(...klines.map((k: any) => k.timestamp)) * 1000

      // Step 2: Fetch triggers from backtest API with time range
      const triggerRes = await fetch(
        `/api/signals/backtest/${signal.id}?symbol=${symbol}&kline_min_ts=${klineMinTs}&kline_max_ts=${klineMaxTs}`
      )
      if (!triggerRes.ok) throw new Error('Failed to fetch trigger data')
      const triggerData = await triggerRes.json()

      // Combine data for preview chart
      // Convert K-line timestamps to milliseconds for consistency
      const formattedKlines = klines.map((k: any) => ({
        timestamp: k.timestamp * 1000,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      }))

      setPreviewData({
        ...triggerData,
        klines: formattedKlines,
        kline_count: formattedKlines.length,
      })
    } catch (err) {
      toast.error('Failed to load preview data')
    } finally {
      setPreviewLoading(false)
    }
  }

  const openPoolPreviewDialog = async (pool: SignalPool, symbol: string = 'BTC') => {
    // Use first signal's time_window or default to 5m, set as default chart timeframe
    const firstSignalId = pool.signal_ids[0]
    const firstSignal = signals.find(s => s.id === firstSignalId)
    const poolTimeWindow = firstSignal?.trigger_condition?.time_window || '5m'
    setChartTimeframe(poolTimeWindow)
    setPreviewPool(pool)
    setPreviewSignal(null)
    setPreviewSymbol(symbol)
    setPreviewDialogOpen(true)
    setPreviewLoading(true)
    setPreviewData(null)

    try {
      // Step 1: Fetch K-lines
      const klineRes = await fetch(
        `/api/market/kline-with-indicators/${symbol}?market=hyperliquid&period=${poolTimeWindow}&count=500`
      )
      if (!klineRes.ok) throw new Error('Failed to fetch K-line data')
      const klineData = await klineRes.json()

      if (!klineData.klines || klineData.klines.length === 0) {
        throw new Error('No K-line data available')
      }

      const klines = klineData.klines
      const klineMinTs = Math.min(...klines.map((k: any) => k.timestamp)) * 1000
      const klineMaxTs = Math.max(...klines.map((k: any) => k.timestamp)) * 1000

      // Step 2: Fetch pool backtest
      const triggerRes = await fetch(
        `/api/signals/pool-backtest/${pool.id}?symbol=${symbol}&kline_min_ts=${klineMinTs}&kline_max_ts=${klineMaxTs}`
      )
      if (!triggerRes.ok) throw new Error('Failed to fetch pool backtest')
      const triggerData = await triggerRes.json()

      const formattedKlines = klines.map((k: any) => ({
        timestamp: k.timestamp * 1000,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      }))

      setPreviewData({
        ...triggerData,
        klines: formattedKlines,
        kline_count: formattedKlines.length,
        isPoolPreview: true,
      })
    } catch (err) {
      toast.error('Failed to load pool preview data')
    } finally {
      setPreviewLoading(false)
    }
  }

  // AI Signal handlers - returns true on success for UI feedback
  const handleAiCreateSignal = async (config: any): Promise<boolean> => {
    try {
      const signalData = {
        signal_name: config.name,
        description: config.description || '',
        trigger_condition: config.trigger_condition,
        enabled: true,
      }
      await createSignal(signalData)
      toast.success(`Signal "${config.name}" created`)
      // Silent refresh - don't close dialog, user may want to create more signals
      const data = await fetchSignals()
      setSignals(data.signals)
      setPools(data.pools)
      return true
    } catch (err) {
      toast.error('Failed to create signal')
      return false
    }
  }

  // AI Signal Pool handler - creates pool from AI-generated config
  const handleAiCreatePool = async (config: any): Promise<boolean> => {
    try {
      const poolConfig = {
        name: config.name,
        symbol: config.symbol,
        description: config.description || '',
        logic: config.logic || 'AND',
        signals: config.signals || [],
      }
      const result = await createPoolFromConfig(poolConfig)
      toast.success(`Signal Pool "${config.name}" created with ${result.signals.length} signals`)
      // Refresh signals and pools
      const data = await fetchSignals()
      setSignals(data.signals)
      setPools(data.pools)
      return true
    } catch (err: any) {
      toast.error(err.message || 'Failed to create signal pool')
      return false
    }
  }

  const handleAiPreviewSignal = async (config: any) => {
    // Create a temporary signal object for preview
    const tempSignal: SignalDefinition = {
      id: 0,
      signal_name: config.name,
      description: config.description || '',
      trigger_condition: config.trigger_condition,
      enabled: true,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    }

    const symbol = config.symbol || 'BTC'
    const tempTimeWindow = config.trigger_condition?.time_window || '5m'
    setChartTimeframe(tempTimeWindow)
    setPreviewSignal(tempSignal)
    setPreviewSymbol(symbol)
    setPreviewDialogOpen(true)
    setPreviewLoading(true)
    setPreviewData(null)

    try {
      // Fetch K-lines
      const klineRes = await fetch(
        `/api/market/kline-with-indicators/${symbol}?market=hyperliquid&period=${tempTimeWindow}&count=500`
      )
      if (!klineRes.ok) throw new Error('Failed to fetch K-line data')
      const klineData = await klineRes.json()

      if (!klineData.klines || klineData.klines.length === 0) {
        throw new Error('No K-line data available')
      }

      const klines = klineData.klines
      const klineMinTs = Math.min(...klines.map((k: any) => k.timestamp)) * 1000
      const klineMaxTs = Math.max(...klines.map((k: any) => k.timestamp)) * 1000

      // Use temp backtest API for preview
      const triggerRes = await fetch('/api/signals/backtest-preview', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol,
          triggerCondition: config.trigger_condition,
          klineMinTs,
          klineMaxTs,
        }),
      })
      if (!triggerRes.ok) throw new Error('Failed to fetch trigger data')
      const triggerData = await triggerRes.json()

      const formattedKlines = klines.map((k: any) => ({
        timestamp: k.timestamp * 1000,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      }))

      setPreviewData({
        ...triggerData,
        klines: formattedKlines,
        kline_count: formattedKlines.length,
      })
    } catch (err) {
      toast.error('Failed to load preview data')
    } finally {
      setPreviewLoading(false)
    }
  }

  // Refresh preview with new chart timeframe (keeps same signal/pool, just changes K-line period)
  const refreshPreviewWithTimeframe = async (newTimeframe: string) => {
    setChartTimeframe(newTimeframe)
    setPreviewLoading(true)

    try {
      // Fetch K-lines with new timeframe
      const klineRes = await fetch(
        `/api/market/kline-with-indicators/${previewSymbol}?market=hyperliquid&period=${newTimeframe}&count=500`
      )
      if (!klineRes.ok) throw new Error('Failed to fetch K-line data')
      const klineData = await klineRes.json()

      if (!klineData.klines || klineData.klines.length === 0) {
        throw new Error('No K-line data available')
      }

      const klines = klineData.klines
      const klineMinTs = Math.min(...klines.map((k: any) => k.timestamp)) * 1000
      const klineMaxTs = Math.max(...klines.map((k: any) => k.timestamp)) * 1000

      // Fetch triggers based on whether it's a pool or signal preview
      let triggerData
      if (previewPool) {
        const triggerRes = await fetch(
          `/api/signals/pool-backtest/${previewPool.id}?symbol=${previewSymbol}&kline_min_ts=${klineMinTs}&kline_max_ts=${klineMaxTs}`
        )
        if (!triggerRes.ok) throw new Error('Failed to fetch pool backtest')
        triggerData = await triggerRes.json()
      } else if (previewSignal) {
        if (previewSignal.id === 0) {
          // Temp signal (AI preview)
          const triggerRes = await fetch('/api/signals/backtest-preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              symbol: previewSymbol,
              triggerCondition: previewSignal.trigger_condition,
              klineMinTs,
              klineMaxTs,
            }),
          })
          if (!triggerRes.ok) throw new Error('Failed to fetch trigger data')
          triggerData = await triggerRes.json()
        } else {
          // Saved signal
          const triggerRes = await fetch(
            `/api/signals/backtest/${previewSignal.id}?symbol=${previewSymbol}&kline_min_ts=${klineMinTs}&kline_max_ts=${klineMaxTs}`
          )
          if (!triggerRes.ok) throw new Error('Failed to fetch trigger data')
          triggerData = await triggerRes.json()
        }
      }

      const formattedKlines = klines.map((k: any) => ({
        timestamp: k.timestamp * 1000,
        open: k.open,
        high: k.high,
        low: k.low,
        close: k.close,
      }))

      setPreviewData({
        ...triggerData,
        klines: formattedKlines,
        kline_count: formattedKlines.length,
        isPoolPreview: !!previewPool,
        chart_timeframe: newTimeframe,
      })
    } catch (err) {
      toast.error('Failed to refresh preview')
    } finally {
      setPreviewLoading(false)
    }
  }

  const openPoolDialog = (pool?: SignalPool) => {
    if (pool) {
      setEditingPool(pool)
      setPoolForm({
        pool_name: pool.pool_name,
        signal_ids: pool.signal_ids,
        symbols: pool.symbols,
        enabled: pool.enabled,
        logic: pool.logic || 'OR',
      })
    } else {
      setEditingPool(null)
      setPoolForm({ pool_name: '', signal_ids: [], symbols: [], enabled: true, logic: 'OR' })
    }
    setPoolDialogOpen(true)
  }

  const handleSavePool = async () => {
    setSavingPool(true)
    try {
      if (editingPool) {
        await updatePool(editingPool.id, poolForm)
        toast.success('Pool updated')
      } else {
        await createPool(poolForm)
        toast.success('Pool created')
      }
      setPoolDialogOpen(false)
      refreshDataSilently()
    } catch (err) {
      toast.error('Failed to save pool')
    } finally {
      setSavingPool(false)
    }
  }

  const handleDeletePool = async (id: number) => {
    if (!confirm('Delete this pool?')) return
    try {
      await deletePool(id)
      toast.success('Pool deleted')
      refreshDataSilently()
    } catch (err) {
      toast.error('Failed to delete pool')
    }
  }

  const toggleSymbol = (symbol: string) => {
    setPoolForm(prev => ({
      ...prev,
      symbols: prev.symbols.includes(symbol)
        ? prev.symbols.filter(s => s !== symbol)
        : [...prev.symbols, symbol]
    }))
  }

  const toggleSignalInPool = (signalId: number) => {
    setPoolForm(prev => ({
      ...prev,
      signal_ids: prev.signal_ids.includes(signalId)
        ? prev.signal_ids.filter(id => id !== signalId)
        : [...prev.signal_ids, signalId]
    }))
  }

  const formatCondition = (cond: TriggerCondition) => {
    const metric = METRICS.find(m => m.value === cond.metric)?.label || cond.metric
    // Handle taker_volume composite signal
    if (cond.metric === 'taker_volume') {
      const dir = (cond as any).direction || 'any'
      const ratio = (cond as any).ratio_threshold || 1.5
      const vol = ((cond as any).volume_threshold || 0).toLocaleString()
      return `${metric} | ${dir.toUpperCase()} ≥${ratio} Vol≥$${vol} (${cond.time_window})`
    }
    const op = OPERATORS.find(o => o.value === cond.operator)?.label || cond.operator
    return `${metric} ${op} ${cond.threshold} (${cond.time_window})`
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64">{t('signals.loading', 'Loading...')}</div>
  }

  return (
    <div className="p-4 space-y-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <div className="flex items-center justify-between gap-4 mb-4">
          <TabsList className="justify-start">
            <TabsTrigger value="signals" className="min-w-[100px]">{t('signals.tabs.signals', 'Signals')}</TabsTrigger>
            <TabsTrigger value="pools" className="min-w-[120px]">{t('signals.tabs.pools', 'Signal Pools')}</TabsTrigger>
            <TabsTrigger value="logs" className="min-w-[120px]">{t('signals.tabs.logs', 'Trigger Logs')}</TabsTrigger>
            <TabsTrigger value="regime" className="min-w-[130px]">{t('signals.tabs.regime', 'Market Regime')}</TabsTrigger>
          </TabsList>
          <p className="text-xs text-amber-600 font-medium flex items-center gap-1">
            <span>⚠️</span>
            <span>{t('signals.mainnetWarning', 'Signal system analyzes Mainnet data only (testnet data unreliable)')}</span>
          </p>
          <div className="flex gap-2">
            <Button onClick={() => openSignalDialog()} size="sm">
              <Plus className="w-4 h-4 mr-2" />{t('signals.newSignal', 'New Signal')}
            </Button>
            <Button onClick={() => openPoolDialog()} size="sm">
              <Plus className="w-4 h-4 mr-2" />{t('signals.newPool', 'New Pool')}
            </Button>
            <Button
              onClick={() => setAiChatOpen(true)}
              size="sm"
              className="bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white border-0 shadow-lg hover:shadow-xl transition-all"
            >
              <Sparkles className="w-4 h-4 mr-2" />{t('signals.aiSetSignal', 'AI Set Signal')}
            </Button>
          </div>
        </div>

        <TabsContent value="signals" className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {signals.map(signal => (
              <Card key={signal.id}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{signal.signal_name}</CardTitle>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" onClick={() => openSignalDialog(signal)}>
                        <Edit className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => handleDeleteSignal(signal.id)}>
                        <Trash2 className="w-4 h-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground mb-2">{signal.description}</p>
                  <p className="text-sm font-mono bg-muted p-2 rounded">
                    {formatCondition(signal.trigger_condition)}
                  </p>
                  <div className="flex items-center justify-between mt-2">
                    <div className="flex items-center gap-2">
                      <span className={`w-2 h-2 rounded-full ${signal.enabled ? 'bg-green-500' : 'bg-gray-400'}`} />
                      <span className="text-xs">{signal.enabled ? t('signals.enabled', 'Enabled') : t('signals.disabled', 'Disabled')}</span>
                    </div>
                    <Button variant="outline" size="sm" onClick={() => openPreviewDialog(signal)}>
                      <Eye className="w-4 h-4 mr-1" />{t('signals.backtest', 'Backtest')}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="pools" className="space-y-4 max-h-[calc(100vh-200px)] overflow-y-auto">
          <div className="grid gap-4 md:grid-cols-2">
            {pools.map(pool => (
              <Card key={pool.id}>
                <CardHeader className="pb-2">
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-lg">{pool.pool_name}</CardTitle>
                    <div className="flex gap-1">
                      <Button variant="ghost" size="sm" onClick={() => openPoolDialog(pool)}>
                        <Edit className="w-4 h-4" />
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => handleDeletePool(pool.id)}>
                        <Trash2 className="w-4 h-4 text-destructive" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div>
                      <span className="text-sm font-medium">{t('signals.symbols', 'Symbols')}: </span>
                      <span className="text-sm">{pool.symbols.join(', ') || 'None'}</span>
                    </div>
                    <div>
                      <span className="text-sm font-medium">{t('signals.tabs.signals', 'Signals')}: </span>
                      <span className="text-sm">
                        {pool.signal_ids.map(id => signals.find(s => s.id === id)?.signal_name).filter(Boolean).join(', ') || 'None'}
                      </span>
                    </div>
                    <div>
                      <span className="text-sm font-medium">{t('signals.logic', 'Logic')}: </span>
                      <span className={`text-sm px-2 py-0.5 rounded ${pool.logic === 'AND' ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'}`}>
                        {pool.logic || 'OR'}
                      </span>
                      <span className="text-xs text-muted-foreground ml-2">
                        {pool.logic === 'AND' ? `(${t('signals.allSignalsTrigger', 'All signals must trigger')})` : `(${t('signals.anySignalTriggers', 'Any signal triggers')})`}
                      </span>
                    </div>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${pool.enabled ? 'bg-green-500' : 'bg-gray-400'}`} />
                        <span className="text-xs">{pool.enabled ? t('signals.enabled', 'Enabled') : t('signals.disabled', 'Disabled')}</span>
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openPoolPreviewDialog(pool, watchlistSymbols[0] || 'BTC')}
                        disabled={pool.signal_ids.length === 0}
                      >
                        <Eye className="w-4 h-4 mr-1" />{t('signals.backtest', 'Backtest')}
                      </Button>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="logs" className="flex-1">
          <Card className="h-full flex flex-col">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="w-5 h-5" />{t('signals.triggerHistory', 'Trigger History')}
              </CardTitle>
            </CardHeader>
            <CardContent className="flex-1 overflow-hidden">
              {logs.length === 0 ? (
                <p className="text-muted-foreground text-center py-8">{t('signals.noTriggers', 'No triggers recorded yet')}</p>
              ) : (
                <ScrollArea className="h-[calc(100vh-280px)]">
                  <div className="space-y-2">
                    {logs.map(log => {
                      const triggerData = log.trigger_value as Record<string, unknown> | null
                      const timestamp = log.triggered_at.endsWith('Z') ? log.triggered_at : log.triggered_at + 'Z'
                      const isPoolTrigger = log.pool_id && triggerData && 'logic' in triggerData
                      const poolName = isPoolTrigger ? pools.find(p => p.id === log.pool_id)?.pool_name : null
                      const signalName = log.signal_id ? signals.find(s => s.id === log.signal_id)?.signal_name : null

                      const formatTriggerDetails = () => {
                        if (!triggerData) return null
                        // Pool trigger (new format)
                        if ('logic' in triggerData && 'signals_triggered' in triggerData) {
                          const logic = triggerData.logic as string
                          const triggeredSignals = triggerData.signals_triggered as Array<{
                            signal_name: string; metric: string; current_value?: number; threshold?: number;
                            direction?: string; volume?: number; volume_threshold?: number;
                          }>
                          return (
                            <div className="space-y-1">
                              <div className="flex items-center gap-2">
                                <span className={`px-1.5 py-0.5 rounded text-xs ${logic === 'AND' ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'}`}>
                                  {logic}
                                </span>
                                <span>Triggered signals:</span>
                              </div>
                              {triggeredSignals.map((s, i) => (
                                <div key={i} className="ml-4 text-xs">
                                  {s.metric === 'taker_volume' ? (
                                    <>• {s.signal_name}: {s.direction?.toUpperCase()} | ratio={s.current_value?.toFixed(2)} (≥{s.threshold}) | vol=${((s.volume || 0) / 1e6).toFixed(2)}M (≥${((s.volume_threshold || 0) / 1e6).toFixed(2)}M)</>
                                  ) : (
                                    <>• {s.signal_name}: {s.metric} = {s.current_value?.toFixed(4)} (threshold: {s.threshold})</>
                                  )}
                                </div>
                              ))}
                            </div>
                          )
                        }
                        // taker_volume composite signal (legacy)
                        if ('direction' in triggerData && 'ratio' in triggerData) {
                          const dir = triggerData.direction as string
                          const ratio = (triggerData.ratio as number)?.toFixed(2)
                          const ratioThreshold = (triggerData.ratio_threshold as number) || 1.5
                          const buy = (triggerData.buy as number) || 0
                          const sell = (triggerData.sell as number) || 0
                          const totalVol = (buy + sell).toLocaleString()
                          const volThreshold = ((triggerData.volume_threshold as number) || 0).toLocaleString()
                          return `${dir.toUpperCase()} | Ratio: ${ratio} (≥${ratioThreshold}) | Vol: $${totalVol} (≥$${volThreshold})`
                        }
                        // Standard signal (legacy)
                        if ('metric' in triggerData && 'value' in triggerData) {
                          const val = (triggerData.value as number)?.toFixed(4)
                          return `${triggerData.metric}: ${val} ${triggerData.operator} ${triggerData.threshold}`
                        }
                        return null
                      }
                      return (
                        <div key={log.id} className="p-3 bg-muted rounded">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <span className="font-medium text-primary">{log.symbol}</span>
                              {isPoolTrigger ? (
                                <span className="text-sm px-2 py-0.5 bg-purple-500/20 text-purple-400 rounded">
                                  Pool: {poolName || `#${log.pool_id}`}
                                </span>
                              ) : (
                                <span className="text-sm">{signalName || `Signal #${log.signal_id}`}</span>
                              )}
                            </div>
                            <span className="text-xs text-muted-foreground">
                              {new Date(timestamp).toLocaleString()}
                            </span>
                          </div>
                          {triggerData && (
                            <div className="text-xs text-muted-foreground mt-1">
                              {formatTriggerDetails()}
                            </div>
                          )}
                          {log.market_regime && (
                            <div className="text-xs mt-1 flex items-center gap-2">
                              <span className={`px-1.5 py-0.5 rounded ${
                                log.market_regime.regime === 'breakout' ? 'bg-green-500/20 text-green-400' :
                                log.market_regime.regime === 'continuation' ? 'bg-blue-500/20 text-blue-400' :
                                log.market_regime.regime === 'absorption' ? 'bg-yellow-500/20 text-yellow-400' :
                                log.market_regime.regime === 'stop_hunt' ? 'bg-red-500/20 text-red-400' :
                                log.market_regime.regime === 'trap' ? 'bg-orange-500/20 text-orange-400' :
                                log.market_regime.regime === 'exhaustion' ? 'bg-purple-500/20 text-purple-400' :
                                'bg-gray-500/20 text-gray-400'
                              }`}>
                                {log.market_regime.regime.toUpperCase()}
                              </span>
                              <span className={log.market_regime.direction === 'bullish' ? 'text-green-400' : 'text-red-400'}>
                                {log.market_regime.direction}
                              </span>
                              <span className="text-muted-foreground">
                                conf: {(log.market_regime.confidence * 100).toFixed(0)}%
                              </span>
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </ScrollArea>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="regime" className="space-y-4">
          <MarketRegimeConfig />
        </TabsContent>
      </Tabs>

      {/* Signal Dialog */}
      <Dialog open={signalDialogOpen} onOpenChange={setSignalDialogOpen}>
        <DialogContent className="max-w-lg">
          <DialogHeader>
            <DialogTitle>{editingSignal ? t('signals.dialog.editSignal', 'Edit Signal') : t('signals.dialog.newSignal', 'New Signal')}</DialogTitle>
            <DialogDescription>{t('signals.dialog.configureSignal', 'Configure when this signal should trigger')}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>{t('signals.dialog.signalNameLabel', 'Signal Name')}</Label>
              <Input
                value={signalForm.signal_name}
                onChange={e => setSignalForm(prev => ({ ...prev, signal_name: e.target.value }))}
                placeholder={t('signals.dialog.signalNamePlaceholder', 'e.g., OI Surge Signal')}
              />
            </div>
            <div>
              <Label>{t('signals.dialog.descriptionLabel', 'Description')}</Label>
              <Input
                value={signalForm.description}
                onChange={e => setSignalForm(prev => ({ ...prev, description: e.target.value }))}
                placeholder={t('signals.dialog.descriptionPlaceholder', 'What market condition does this signal detect?')}
              />
            </div>
            <div>
              <Label>{t('signals.dialog.metricLabel', 'Metric')}</Label>
              <Select value={signalForm.metric} onValueChange={v => setSignalForm(prev => ({ ...prev, metric: v }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {METRICS.map(m => <SelectItem key={m.value} value={m.value}>{m.label}</SelectItem>)}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {METRICS.find(m => m.value === signalForm.metric)?.desc}
              </p>
            </div>
            {signalForm.metric === 'taker_volume' ? (
              /* Composite signal UI for taker_volume */
              <div className="space-y-4 p-3 bg-blue-500/10 rounded-lg border border-blue-500/30">
                <div className="text-xs font-medium text-blue-400">{t('signals.dialog.compositeConfig', 'Composite Signal Configuration')}</div>
                <div>
                  <Label>{t('signals.dialog.directionLabel', 'Direction')}</Label>
                  <Select value={signalForm.direction} onValueChange={v => setSignalForm(prev => ({ ...prev, direction: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {TAKER_DIRECTIONS.map(d => <SelectItem key={d.value} value={d.value}>{d.label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    {TAKER_DIRECTIONS.find(d => d.value === signalForm.direction)?.desc}
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <Label>{t('signals.dialog.ratioThreshold', 'Ratio Threshold')}</Label>
                    <Input
                      type="number"
                      step="0.1"
                      min="1.01"
                      value={signalForm.ratio_threshold}
                      onChange={e => setSignalForm(prev => ({ ...prev, ratio_threshold: parseFloat(e.target.value) || 1.5 }))}
                    />
                    <p className="text-xs text-muted-foreground mt-1">{t('signals.dialog.ratioThresholdDesc', 'Multiplier (e.g., 1.5 = 50% more). Symmetric for buy/sell.')}</p>
                  </div>
                  <div>
                    <Label>{t('signals.dialog.volumeThreshold', 'Volume Threshold')}</Label>
                    <Input
                      type="number"
                      step="1000"
                      min="0"
                      value={signalForm.volume_threshold}
                      onChange={e => setSignalForm(prev => ({ ...prev, volume_threshold: parseFloat(e.target.value) || 0 }))}
                    />
                    <p className="text-xs text-muted-foreground mt-1">{t('signals.dialog.volumeThresholdDesc', 'Min volume (USD)')}</p>
                  </div>
                </div>
              </div>
            ) : (
              /* Standard signal UI */
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Label>{t('signals.dialog.operatorLabel', 'Operator')}</Label>
                  <Select value={signalForm.operator} onValueChange={v => setSignalForm(prev => ({ ...prev, operator: v }))}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {OPERATORS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground mt-1">
                    {OPERATORS.find(o => o.value === signalForm.operator)?.desc}
                  </p>
                </div>
                <div>
                  <Label>{t('signals.dialog.thresholdLabel', 'Threshold')}</Label>
                  <Input
                    type="number"
                    step="0.1"
                    value={signalForm.threshold}
                    onChange={e => setSignalForm(prev => ({ ...prev, threshold: parseFloat(e.target.value) || 0 }))}
                  />
                  <p className="text-xs text-muted-foreground mt-1">{t('signals.dialog.thresholdDesc', 'Value to compare against')}</p>
                </div>
              </div>
            )}
            <div>
              <Label>{t('signals.dialog.timeWindowLabel', 'Time Window')}</Label>
              <Select value={signalForm.time_window} onValueChange={v => setSignalForm(prev => ({ ...prev, time_window: v }))}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {TIME_WINDOWS.map(tw => <SelectItem key={tw.value} value={tw.value}>{tw.label}</SelectItem>)}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {TIME_WINDOWS.find(tw => tw.value === signalForm.time_window)?.desc}
              </p>
            </div>

            {/* Statistical Analysis Preview */}
            <div className="p-3 bg-muted/50 rounded-lg border">
              <div className="flex items-center gap-2 mb-2">
                <span className="text-sm font-medium">{t('signals.dialog.statisticalAnalysis', 'Statistical Analysis')}</span>
                <Select value={analysisSymbol} onValueChange={setAnalysisSymbol}>
                  <SelectTrigger className="w-24 h-7 text-xs">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {watchlistSymbols.length > 0 ? (
                      watchlistSymbols.map(sym => <SelectItem key={sym} value={sym}>{sym}</SelectItem>)
                    ) : (
                      <SelectItem value="BTC">BTC</SelectItem>
                    )}
                  </SelectContent>
                </Select>
                {watchlistSymbols.length === 0 && (
                  <span className="text-xs text-muted-foreground">{t('signals.dialog.addSymbolsHint', '(Add symbols in AI Trader)')}</span>
                )}
              </div>
              {analysisLoading ? (
                <p className="text-xs text-muted-foreground">{t('signals.dialog.loadingAnalysis', 'Loading analysis...')}</p>
              ) : metricAnalysis?.status === 'ok' && metricAnalysis.metric === signalForm.metric ? (
                signalForm.metric === 'taker_volume' && (metricAnalysis as any).ratio_statistics ? (
                  /* taker_volume composite analysis */
                  <div className="space-y-3">
                    <p className="text-xs text-muted-foreground">
                      Based on {metricAnalysis.sample_count} samples over {metricAnalysis.time_range_hours.toFixed(1)} hours
                    </p>
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-2 bg-background rounded border">
                        <div className="text-xs font-medium mb-1">Ratio Multiplier</div>
                        <div className="text-xs text-muted-foreground mb-2">
                          Log range: {(metricAnalysis as any).ratio_statistics?.min?.toFixed(2)} ~ {(metricAnalysis as any).ratio_statistics?.max?.toFixed(2)} (0=balanced)
                        </div>
                        <div className="flex flex-wrap gap-1">
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, ratio_threshold: (metricAnalysis as any).suggestions?.ratio?.aggressive }))} className="text-xs px-1.5 py-0.5 bg-muted border rounded hover:bg-accent">
                            {(metricAnalysis as any).suggestions?.ratio?.aggressive?.toFixed(2)}x
                          </button>
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, ratio_threshold: (metricAnalysis as any).suggestions?.ratio?.moderate }))} className="text-xs px-1.5 py-0.5 bg-primary/10 border border-primary rounded hover:bg-primary/20">
                            {(metricAnalysis as any).suggestions?.ratio?.moderate?.toFixed(2)}x ★
                          </button>
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, ratio_threshold: (metricAnalysis as any).suggestions?.ratio?.conservative }))} className="text-xs px-1.5 py-0.5 bg-muted border rounded hover:bg-accent">
                            {(metricAnalysis as any).suggestions?.ratio?.conservative?.toFixed(2)}x
                          </button>
                        </div>
                      </div>
                      <div className="p-2 bg-background rounded border">
                        <div className="text-xs font-medium mb-1">Volume (USD)</div>
                        <div className="text-xs text-muted-foreground mb-2">
                          Range: {((metricAnalysis as any).volume_statistics?.min / 1000)?.toFixed(0)}K ~ {((metricAnalysis as any).volume_statistics?.max / 1000)?.toFixed(0)}K
                        </div>
                        <div className="flex flex-wrap gap-1">
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, volume_threshold: (metricAnalysis as any).suggestions?.volume?.low }))} className="text-xs px-1.5 py-0.5 bg-muted border rounded hover:bg-accent">
                            {((metricAnalysis as any).suggestions?.volume?.low / 1000)?.toFixed(0)}K
                          </button>
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, volume_threshold: (metricAnalysis as any).suggestions?.volume?.medium }))} className="text-xs px-1.5 py-0.5 bg-primary/10 border border-primary rounded hover:bg-primary/20">
                            {((metricAnalysis as any).suggestions?.volume?.medium / 1000)?.toFixed(0)}K ★
                          </button>
                          <button type="button" onClick={() => setSignalForm(prev => ({ ...prev, volume_threshold: (metricAnalysis as any).suggestions?.volume?.high }))} className="text-xs px-1.5 py-0.5 bg-muted border rounded hover:bg-accent">
                            {((metricAnalysis as any).suggestions?.volume?.high / 1000)?.toFixed(0)}K
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : metricAnalysis.suggestions ? (
                  /* Standard metric analysis with suggestions */
                  <div className="space-y-2">
                    <p className="text-xs text-muted-foreground">
                      Based on {metricAnalysis.sample_count} samples over {metricAnalysis.time_range_hours.toFixed(1)} hours
                    </p>
                    {metricAnalysis.warning && (
                      <p className="text-xs text-yellow-600">{metricAnalysis.warning}</p>
                    )}
                    <div className="text-xs">
                      <span className="text-muted-foreground">Range: </span>
                      {signalForm.metric === 'funding'
                        ? `${metricAnalysis.statistics?.min.toFixed(1)} ~ ${metricAnalysis.statistics?.max.toFixed(1)}`
                        : `${metricAnalysis.statistics?.min.toFixed(4)} ~ ${metricAnalysis.statistics?.max.toFixed(4)}`
                      }
                    </div>
                    <div className="text-xs font-medium mt-2">{t('signals.dialog.suggestedThresholds', 'Suggested thresholds:')}</div>
                    <div className="flex flex-wrap gap-2 mt-1">
                      <button
                        type="button"
                        onClick={() => setSignalForm(prev => ({ ...prev, threshold: metricAnalysis.suggestions!.aggressive.threshold }))}
                        className="text-xs px-2 py-1 bg-background border rounded hover:bg-accent"
                        title={metricAnalysis.suggestions.aggressive.description}
                      >
                        {t('signals.dialog.aggressive', 'Aggressive')} {signalForm.metric === 'funding'
                          ? metricAnalysis.suggestions.aggressive.threshold.toFixed(1)
                          : metricAnalysis.suggestions.aggressive.threshold.toFixed(4)}
                        {(metricAnalysis.suggestions.aggressive as any).multiplier && ` (${(metricAnalysis.suggestions.aggressive as any).multiplier}x)`}
                      </button>
                      <button
                        type="button"
                        onClick={() => setSignalForm(prev => ({ ...prev, threshold: metricAnalysis.suggestions!.moderate.threshold }))}
                        className="text-xs px-2 py-1 bg-primary/10 border border-primary rounded hover:bg-primary/20"
                        title={metricAnalysis.suggestions.moderate.description}
                      >
                        {t('signals.dialog.moderate', 'Moderate')} {signalForm.metric === 'funding'
                          ? metricAnalysis.suggestions.moderate.threshold.toFixed(1)
                          : metricAnalysis.suggestions.moderate.threshold.toFixed(4)}
                        {(metricAnalysis.suggestions.moderate as any).multiplier && ` (${(metricAnalysis.suggestions.moderate as any).multiplier}x)`} ★
                      </button>
                      <button
                        type="button"
                        onClick={() => setSignalForm(prev => ({ ...prev, threshold: metricAnalysis.suggestions!.conservative.threshold }))}
                        className="text-xs px-2 py-1 bg-background border rounded hover:bg-accent"
                        title={metricAnalysis.suggestions.conservative.description}
                      >
                        {t('signals.dialog.conservative', 'Conservative')} {signalForm.metric === 'funding'
                          ? metricAnalysis.suggestions.conservative.threshold.toFixed(1)
                          : metricAnalysis.suggestions.conservative.threshold.toFixed(4)}
                        {(metricAnalysis.suggestions.conservative as any).multiplier && ` (${(metricAnalysis.suggestions.conservative as any).multiplier}x)`}
                      </button>
                    </div>
                  </div>
                ) : (
                  /* Fallback when no suggestions available */
                  <p className="text-xs text-muted-foreground">{t('signals.dialog.analysisDataMismatch', 'Analysis data format mismatch. Please reselect metric.')}</p>
                )
              ) : metricAnalysis?.status === 'insufficient_data' ? (
                <p className="text-xs text-yellow-600">{metricAnalysis.message}</p>
              ) : (
                <p className="text-xs text-muted-foreground">{t('signals.dialog.unableToLoadAnalysis', 'Unable to load analysis')}</p>
              )}
            </div>

            <div className="flex items-center gap-2">
              <Switch checked={signalForm.enabled} onCheckedChange={v => setSignalForm(prev => ({ ...prev, enabled: v }))} />
              <Label>{t('signals.dialog.enabledLabel', 'Enabled')}</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setSignalDialogOpen(false)} disabled={savingSignal}>{t('signals.dialog.cancel', 'Cancel')}</Button>
            <Button onClick={handleSaveSignal} disabled={savingSignal}>
              {savingSignal ? t('signals.dialog.saving', 'Saving...') : t('signals.dialog.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Pool Dialog */}
      <Dialog open={poolDialogOpen} onOpenChange={setPoolDialogOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>{editingPool ? t('signals.dialog.editPool', 'Edit Pool') : t('signals.dialog.newPool', 'New Pool')}</DialogTitle>
            <DialogDescription>{t('signals.dialog.configurePool', 'Configure signal pool')}</DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div>
              <Label>{t('signals.dialog.poolNameLabel', 'Pool Name')}</Label>
              <Input
                value={poolForm.pool_name}
                onChange={e => setPoolForm(prev => ({ ...prev, pool_name: e.target.value }))}
                placeholder={t('signals.dialog.poolNamePlaceholder', 'e.g., BTC Momentum Pool')}
              />
            </div>
            <div>
              <Label>{t('signals.dialog.symbolsLabel', 'Symbols')}</Label>
              <div className="flex flex-wrap gap-2 mt-2">
                {watchlistSymbols.length > 0 ? (
                  watchlistSymbols.map(symbol => (
                    <Button
                      key={symbol}
                      variant={poolForm.symbols.includes(symbol) ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => toggleSymbol(symbol)}
                    >
                      {symbol}
                    </Button>
                  ))
                ) : (
                  <p className="text-sm text-muted-foreground">{t('signals.dialog.noSymbolsInWatchlist', 'No symbols in watchlist. Configure watchlist first.')}</p>
                )}
              </div>
            </div>
            <div>
              <Label>{t('signals.dialog.signalsLabel', 'Signals')}</Label>
              <div className="space-y-2 mt-2 max-h-40 overflow-y-auto">
                {signals.map(signal => (
                  <div key={signal.id} className="flex items-center gap-2">
                    <Switch
                      checked={poolForm.signal_ids.includes(signal.id)}
                      onCheckedChange={() => toggleSignalInPool(signal.id)}
                    />
                    <span className="text-sm">{signal.signal_name}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <Label>{t('signals.dialog.triggerLogicLabel', 'Trigger Logic')}</Label>
              <Select value={poolForm.logic} onValueChange={(v: 'OR' | 'AND') => setPoolForm(prev => ({ ...prev, logic: v }))}>
                <SelectTrigger className="mt-2">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="OR">{t('signals.dialog.orLogic', 'OR - Any signal triggers pool')}</SelectItem>
                  <SelectItem value="AND">{t('signals.dialog.andLogic', 'AND - All signals must trigger')}</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                {poolForm.logic === 'AND'
                  ? t('signals.dialog.andLogicDesc', 'Pool triggers only when ALL selected signals meet their conditions simultaneously')
                  : t('signals.dialog.orLogicDesc', 'Pool triggers when ANY selected signal meets its condition')}
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Switch checked={poolForm.enabled} onCheckedChange={v => setPoolForm(prev => ({ ...prev, enabled: v }))} />
              <Label>{t('signals.dialog.enabledLabel', 'Enabled')}</Label>
            </div>
          </div>
          <DialogFooter>
            <Button variant="outline" onClick={() => setPoolDialogOpen(false)} disabled={savingPool}>{t('signals.dialog.cancel', 'Cancel')}</Button>
            <Button onClick={handleSavePool} disabled={savingPool}>
              {savingPool ? t('signals.dialog.saving', 'Saving...') : t('signals.dialog.save', 'Save')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Signal/Pool Preview Dialog */}
      <Dialog open={previewDialogOpen} onOpenChange={setPreviewDialogOpen}>
        <DialogContent className="w-[1200px] max-w-[95vw] h-[860px] max-h-[95vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>
              {previewPool ? t('signals.preview.poolPreview', { name: previewPool.pool_name, defaultValue: `Pool Preview: ${previewPool.pool_name}` }) : t('signals.preview.signalPreview', { name: previewSignal?.signal_name, defaultValue: `Signal Preview: ${previewSignal?.signal_name}` })}
            </DialogTitle>
            <DialogDescription>
              {previewPool
                ? t('signals.preview.poolBacktestDesc', { logic: previewPool.logic || 'OR', defaultValue: `Historical backtest showing combined triggers (${previewPool.logic || 'OR'} logic)` })
                : t('signals.preview.signalBacktestDesc', 'Historical backtest showing where this signal would have triggered')}
            </DialogDescription>
          </DialogHeader>

          {previewLoading ? (
            <div className="flex items-center justify-center h-[500px] gap-3">
              <PacmanLoader className="w-16 h-8" />
              <span className="text-muted-foreground">{t('signals.preview.loadingPreview', 'Loading preview data...')}</span>
            </div>
          ) : previewData?.error ? (
            <div className="flex items-center justify-center h-[500px]">
              <div className="text-center text-destructive">
                <p className="font-medium">{t('signals.preview.previewError', 'Preview Error')}</p>
                <p className="text-sm mt-2">{previewData.error}</p>
              </div>
            </div>
          ) : previewData?.klines ? (
            <div className="space-y-4">
              {/* Signal Info */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                <div className="bg-muted p-3 rounded">
                  <div className="text-muted-foreground">{t('signals.preview.symbol', 'Symbol')}</div>
                  <div className="font-medium">{previewData.symbol}</div>
                </div>
                <div className="bg-muted p-3 rounded">
                  <div className="text-muted-foreground">{t('signals.preview.timeWindow', 'Time Window')}</div>
                  <div className="font-medium">{previewData.time_window}</div>
                </div>
                <div className="bg-muted p-3 rounded">
                  <div className="text-muted-foreground">{t('signals.preview.klines', 'K-lines')}</div>
                  <div className="font-medium">{previewData.kline_count}</div>
                </div>
                <div className="bg-muted p-3 rounded">
                  <div className="text-muted-foreground">{t('signals.preview.triggers', 'Triggers')}</div>
                  <div className="flex items-center gap-2">
                    <span className="font-medium text-yellow-500">{previewData.trigger_count}</span>
                    {previewData.trigger_count > 0 && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-6 text-xs px-2"
                        disabled={regimeLoading}
                        onClick={checkMarketRegime}
                      >
                        {regimeLoading ? t('signals.preview.checking', 'Checking...') : t('signals.preview.checkRegime', 'Check Regime')}
                      </Button>
                    )}
                  </div>
                </div>
              </div>

              {/* Condition Display */}
              <div className="bg-muted p-3 rounded text-sm">
                {previewData.isPoolPreview ? (
                  <div className="space-y-1">
                    <div>
                      <span className="text-muted-foreground">{t('signals.preview.logic', 'Logic')}: </span>
                      <span className={`px-2 py-0.5 rounded ${previewData.logic === 'AND' ? 'bg-blue-500/20 text-blue-400' : 'bg-green-500/20 text-green-400'}`}>
                        {previewData.logic || 'OR'}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">{t('signals.preview.signals', 'Signals')}: </span>
                      <span className="font-mono">
                        {Object.values(previewData.signal_names || {}).join(', ')}
                      </span>
                    </div>
                  </div>
                ) : (
                  <>
                    <span className="text-muted-foreground">{t('signals.preview.condition', 'Condition')}: </span>
                    <span className="font-mono">
                      {previewData.condition?.metric} {previewData.condition?.operator} {previewData.condition?.threshold}
                    </span>
                  </>
                )}
              </div>

              {/* Chart */}
              <div className="border rounded-lg overflow-hidden">
                <SignalPreviewChart
                  klines={previewData.klines}
                  triggers={previewData.triggers || []}
                  timeWindow={chartTimeframe}
                />
              </div>

              {/* Chart Timeframe Selector */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-muted-foreground">{t('signals.preview.chartTimeframe', 'Chart timeframe:')}</span>
                {['1m', '3m', '5m', '15m', '30m', '1h', '4h'].map(tf => (
                  <Button
                    key={tf}
                    variant={chartTimeframe === tf ? 'default' : 'outline'}
                    size="sm"
                    disabled={previewLoading}
                    onClick={() => refreshPreviewWithTimeframe(tf)}
                  >
                    {tf}
                  </Button>
                ))}
                <span className="text-xs text-muted-foreground ml-2">
                  ({t('signals.preview.signal', 'Signal')}: {previewData.time_window || '5m'})
                </span>
                <span className="text-xs text-yellow-500 ml-2">
                  {t('signals.preview.largerTimeframeNote', 'Note: Larger timeframes require longer calculation time')}
                </span>
              </div>

              {/* Symbol Selector */}
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-sm text-muted-foreground">{t('signals.preview.changeSymbol', 'Change symbol:')}</span>
                {watchlistSymbols.length > 0 ? (
                  watchlistSymbols.map(sym => (
                    <Button
                      key={sym}
                      variant={previewSymbol === sym ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => {
                        if (previewPool) {
                          openPoolPreviewDialog(previewPool, sym)
                        } else if (previewSignal) {
                          openPreviewDialog(previewSignal, sym)
                        }
                      }}
                    >
                      {sym}
                    </Button>
                  ))
                ) : (
                  <span className="text-sm text-muted-foreground italic">No symbols in Watchlist</span>
                )}
                <span className="text-xs text-muted-foreground ml-2">
                  (Manage symbols in AI Trader page)
                </span>
              </div>
            </div>
          ) : (
            <div className="text-center text-muted-foreground py-8">
              No data available
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* AI Signal Chat Modal */}
      <AiSignalChatModal
        open={aiChatOpen}
        onOpenChange={setAiChatOpen}
        onCreateSignal={handleAiCreateSignal}
        onCreatePool={handleAiCreatePool}
        onPreviewSignal={handleAiPreviewSignal}
        accounts={accounts}
        accountsLoading={accountsLoading}
      />
    </div>
  )
}
