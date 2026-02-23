/**
 * HyperAiPage - Independent page for Hyper AI (three-column layout)
 * Left: Conversation list
 * Center: Chat area
 * Right: Config panel
 */
import { useState, useEffect, useRef, memo } from 'react'
import { useTranslation } from 'react-i18next'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { ScrollArea } from '@/components/ui/scroll-area'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import PacmanLoader from '@/components/ui/pacman-loader'
import {
  Plus,
  Send,
  Settings,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  Loader2,
  Bot,
  Pencil,
  X,
  CheckCircle2,
  AlertCircle,
  User,
  Wrench,
  Play
} from 'lucide-react'

interface Conversation {
  id: number
  title: string
  message_count: number
  updated_at: string
}

interface ToolCallEntry {
  type: 'tool_call' | 'tool_result' | 'reasoning'
  name?: string
  args?: Record<string, unknown>
  result?: string
  content?: string
}

// API format for tool_calls_log from database
interface ToolCallLogEntry {
  tool: string
  args: Record<string, unknown>
  result: string
}

interface Message {
  id?: number
  role: 'user' | 'assistant'
  content: string
  reasoning_snapshot?: string
  tool_calls_log?: string
  is_complete?: boolean
  interrupt_reason?: string
  created_at?: string
  // Streaming state
  isStreaming?: boolean
  statusText?: string
  toolCalls?: ToolCallEntry[]
  isInterrupted?: boolean
  interruptedRound?: number
}

interface CompressionPoint {
  message_id: number
  summary: string
  compressed_at: string
}

interface TokenUsage {
  current_tokens: number
  max_tokens: number
  usage_ratio: number
  show_warning: boolean
}

interface LLMProvider {
  id: string
  name: string
  models: string[]
  base_url?: string
}

// LLM Config Modal component
function LLMConfigModal({
  open,
  onClose,
  providers,
  currentProfile,
  onSaved
}: {
  open: boolean
  onClose: () => void
  providers: LLMProvider[]
  currentProfile: any
  onSaved: () => void
}) {
  const { t } = useTranslation()
  const [selectedProvider, setSelectedProvider] = useState(currentProfile?.llm_provider || '')
  const [apiKey, setApiKey] = useState('')
  const [modelInput, setModelInput] = useState(currentProfile?.llm_model || '')
  const [customBaseUrl, setCustomBaseUrl] = useState(currentProfile?.llm_base_url || '')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const currentProvider = providers.find(p => p.id === selectedProvider)

  useEffect(() => {
    if (open) {
      setSelectedProvider(currentProfile?.llm_provider || '')
      setModelInput(currentProfile?.llm_model || '')
      setCustomBaseUrl(currentProfile?.llm_base_url || '')
      setApiKey('')
      setError('')
      setSuccess(false)
    }
  }, [open, currentProfile])

  // When provider changes, set default model if current model is empty
  useEffect(() => {
    if (selectedProvider && !modelInput) {
      const provider = providers.find(p => p.id === selectedProvider)
      if (provider && provider.models.length > 0) {
        setModelInput(provider.models[0])
      }
    }
  }, [selectedProvider])

  const handleSave = async () => {
    if (!selectedProvider || !apiKey) {
      setError(t('hyperAi.onboarding.fillRequired', 'Please fill in all required fields'))
      return
    }

    if (selectedProvider === 'custom' && !customBaseUrl) {
      setError(t('hyperAi.onboarding.baseUrlRequired', 'Base URL is required for custom provider'))
      return
    }

    setSaving(true)
    setError('')

    try {
      const res = await fetch('/api/hyper-ai/profile/llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider,
          api_key: apiKey,
          model: modelInput,
          base_url: selectedProvider === 'custom' ? customBaseUrl : undefined
        })
      })

      if (!res.ok) {
        const errData = await res.json()
        throw new Error(errData.detail || 'Connection test failed')
      }

      setSuccess(true)
      setTimeout(() => {
        onSaved()
        onClose()
      }, 800)
    } catch (e: any) {
      setError(e.message || 'Failed to save')
    } finally {
      setSaving(false)
    }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="w-full max-w-md bg-background rounded-lg shadow-xl p-6 space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">{t('hyperAi.configTitle', 'Hyper AI Config')}</h2>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="w-4 h-4" />
          </Button>
        </div>

        <div className="space-y-4">
          <div className="space-y-2">
            <Label>{t('hyperAi.onboarding.provider', 'AI Provider')}</Label>
            <Select value={selectedProvider} onValueChange={(v) => { setSelectedProvider(v); setModelInput('') }}>
              <SelectTrigger>
                <SelectValue placeholder={t('hyperAi.onboarding.selectProvider', 'Select provider')} />
              </SelectTrigger>
              <SelectContent>
                {providers.map(p => (
                  <SelectItem key={p.id} value={p.id}>{p.name}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {selectedProvider === 'custom' && (
            <div className="space-y-2">
              <Label>{t('hyperAi.onboarding.baseUrl', 'Base URL')}</Label>
              <Input
                value={customBaseUrl}
                onChange={e => setCustomBaseUrl(e.target.value)}
                placeholder="https://api.example.com/v1"
              />
            </div>
          )}

          <div className="space-y-2">
            <Label>{t('hyperAi.onboarding.apiKey', 'API Key')}</Label>
            <Input
              type="password"
              value={apiKey}
              onChange={e => setApiKey(e.target.value)}
              placeholder={currentProfile?.llm_configured ? t('hyperAi.onboarding.apiKeyConfigured', 'Enter new API key to update') : 'sk-...'}
            />
          </div>

          {selectedProvider && (
            <div className="space-y-2">
              <Label>{t('hyperAi.onboarding.model', 'Model')}</Label>
              <div className="flex gap-1">
                <Input
                  value={modelInput}
                  onChange={e => setModelInput(e.target.value)}
                  placeholder={t('hyperAi.onboarding.modelPlaceholder', 'Enter or select model')}
                  className="flex-1"
                />
                {currentProvider && currentProvider.models.length > 0 && (
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="icon" className="shrink-0">
                        <ChevronDown className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="max-h-60 overflow-y-auto">
                      {currentProvider.models.map(m => (
                        <DropdownMenuItem key={m} onClick={() => setModelInput(m)}>
                          {m}
                        </DropdownMenuItem>
                      ))}
                    </DropdownMenuContent>
                  </DropdownMenu>
                )}
              </div>
            </div>
          )}
        </div>

        {error && (
          <div className="flex items-center gap-2 text-destructive text-sm">
            <AlertCircle className="w-4 h-4 flex-shrink-0" />
            <span className="break-all">{error}</span>
          </div>
        )}

        {success && (
          <div className="flex items-center gap-2 text-green-600 text-sm">
            <CheckCircle2 className="w-4 h-4" />
            {t('hyperAi.onboarding.connectionSuccess', 'Connection successful!')}
          </div>
        )}

        <div className="flex gap-3 pt-2">
          <Button variant="outline" onClick={onClose} className="flex-1">
            {t('common.cancel', 'Cancel')}
          </Button>
          <Button onClick={handleSave} disabled={!selectedProvider || !apiKey || saving} className="flex-1">
            {saving && <Loader2 className="w-4 h-4 animate-spin mr-2" />}
            {saving ? t('hyperAi.onboarding.testing', 'Testing...') : t('common.save', 'Save')}
          </Button>
        </div>
      </div>
    </div>
  )
}

// Welcome message component
function WelcomeMessage({ nickname, t }: { nickname?: string; t: any }) {
  const greeting = nickname
    ? t('hyperAi.welcomeWithName', { name: nickname, defaultValue: `你好，${nickname}！我是 Hyper AI，你的专属交易助手。` })
    : t('hyperAi.welcomeNoName', '你好！我是 Hyper AI，Hyper Alpha Arena 的智能助手。')

  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
        <Bot className="w-8 h-8 text-primary" />
      </div>
      <p className="text-lg mb-4">{greeting}</p>
      <div className="text-sm text-muted-foreground space-y-1 max-w-md">
        <p>{t('hyperAi.welcomeCapabilities', '我可以帮你：')}</p>
        <ul className="text-left list-disc list-inside space-y-1 mt-2">
          <li>{t('hyperAi.capability1', '了解系统功能和使用方法')}</li>
          <li>{t('hyperAi.capability2', '生成和优化 AI 交易策略')}</li>
          <li>{t('hyperAi.capability3', '管理 AI 交易员和钱包配置')}</li>
          <li>{t('hyperAi.capability4', '分析市场数据和交易表现')}</li>
        </ul>
        <p className="mt-4">{t('hyperAi.welcomePrompt', '有什么想了解的，直接问我就行。')}</p>
      </div>
    </div>
  )
}

export default function HyperAiPage() {
  const { t, i18n } = useTranslation()
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConvId, setCurrentConvId] = useState<number | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [compressionPoints, setCompressionPoints] = useState<CompressionPoint[]>([])
  const [tokenUsage, setTokenUsage] = useState<TokenUsage | null>(null)
  const [inputValue, setInputValue] = useState('')
  const [sending, setSending] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [providers, setProviders] = useState<LLMProvider[]>([])
  const [profile, setProfile] = useState<any>(null)
  const [nickname, setNickname] = useState<string>('')
  const [showConfig, setShowConfig] = useState(true)
  const [showConfigModal, setShowConfigModal] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Get current language
  const currentLang = i18n.language?.startsWith('zh') ? 'zh' : 'en'

  useEffect(() => {
    fetchConversations()
    fetchProviders()
    fetchProfile()
  }, [])

  useEffect(() => {
    // Don't fetch messages while sending - it would overwrite the streaming message
    if (currentConvId && !sending) {
      fetchMessages(currentConvId)
    }
  }, [currentConvId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  const fetchConversations = async () => {
    try {
      const res = await fetch('/api/hyper-ai/conversations')
      const data = await res.json()
      setConversations(data.conversations || [])
    } catch (e) {
      console.error('Failed to fetch conversations:', e)
    }
  }

  const fetchMessages = async (convId: number) => {
    try {
      const res = await fetch(`/api/hyper-ai/conversations/${convId}/messages`)
      const data = await res.json()
      setMessages(data.messages || [])
      setCompressionPoints(data.compression_points || [])
      setTokenUsage(data.token_usage || null)
    } catch (e) {
      console.error('Failed to fetch messages:', e)
    }
  }

  const fetchProviders = async () => {
    try {
      const res = await fetch('/api/hyper-ai/providers')
      const data = await res.json()
      setProviders(data.providers || [])
    } catch (e) {
      console.error('Failed to fetch providers:', e)
    }
  }

  const fetchProfile = async () => {
    try {
      const res = await fetch('/api/hyper-ai/profile')
      const data = await res.json()
      setProfile(data)
      if (data.nickname) {
        setNickname(data.nickname)
      }
    } catch (e) {
      console.error('Failed to fetch profile:', e)
    }
  }

  const handleNewConversation = () => {
    // Lazy creation: just clear current state, don't create in DB yet
    setCurrentConvId(null)
    setMessages([])
    setCompressionPoints([])
    setTokenUsage(null)
  }

  const handleSend = async () => {
    if (!inputValue.trim() || sending) return

    const userMessage = inputValue.trim()
    setInputValue('')
    setSending(true)
    setStreamingContent('')

    // Add user message and placeholder assistant message
    const tempAssistantId = Date.now()
    setMessages(prev => [
      ...prev,
      { role: 'user', content: userMessage },
      {
        role: 'assistant',
        content: '',
        isStreaming: true,
        statusText: t('hyperAi.connecting', 'Connecting...'),
        toolCalls: []
      }
    ])

    try {
      const res = await fetch('/api/hyper-ai/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          conversation_id: currentConvId,
          lang: currentLang
        })
      })

      const data = await res.json()
      if (data.task_id) {
        // Poll for streaming response
        pollTaskResponse(data.task_id, data.conversation_id)
        if (!currentConvId) {
          setCurrentConvId(data.conversation_id)
        }
      }
    } catch (e) {
      console.error('Failed to send message:', e)
      // Remove placeholder on error
      setMessages(prev => prev.slice(0, -1))
      setSending(false)
    }
  }

  const pollTaskResponse = async (taskId: string, convId: number) => {
    let content = ''
    let reasoning = ''
    let toolCalls: ToolCallEntry[] = []
    let offset = 0
    let isInterrupted = false
    let interruptedRound = 0

    // Update currentConvId immediately if not set
    if (!currentConvId && convId) {
      setCurrentConvId(convId)
    }

    try {
      while (true) {
        await new Promise(resolve => setTimeout(resolve, 300)) // Poll every 300ms

        const pollResponse = await fetch(`/api/ai-stream/${taskId}?offset=${offset}`)
        if (!pollResponse.ok) {
          console.error('Failed to poll task')
          break
        }

        const pollData = await pollResponse.json()
        const { chunks, status, next_offset } = pollData

        // Process chunks
        for (const chunk of chunks) {
          const eventType = chunk.event_type
          const data = chunk.data

          if (eventType === 'content' && data.text) {
            content += data.text
            setStreamingContent(content)
            // Update status to empty when content starts
            setMessages(prev => prev.map((m, idx) =>
              idx === prev.length - 1 && m.isStreaming
                ? { ...m, content, statusText: '' }
                : m
            ))
          } else if (eventType === 'reasoning' && data.content) {
            // Real-time reasoning display (same as AI Program)
            reasoning += data.content
            setMessages(prev => prev.map((m, idx) =>
              idx === prev.length - 1 && m.isStreaming
                ? {
                    ...m,
                    toolCalls: [...(m.toolCalls || []), { type: 'reasoning', content: data.content as string }],
                  }
                : m
            ))
          } else if (eventType === 'tool_call' && data.name) {
            toolCalls.push({ type: 'tool_call', name: data.name, args: data.args || {} })
            setMessages(prev => prev.map((m, idx) =>
              idx === prev.length - 1 && m.isStreaming
                ? {
                    ...m,
                    statusText: `${t('hyperAi.calling', 'Calling')} ${data.name}...`,
                    toolCalls: [...(m.toolCalls || []), { type: 'tool_call', name: data.name, args: data.args }]
                  }
                : m
            ))
          } else if (eventType === 'tool_result' && data.name) {
            toolCalls.push({ type: 'tool_result', name: data.name, result: data.result })
            setMessages(prev => prev.map((m, idx) =>
              idx === prev.length - 1 && m.isStreaming
                ? {
                    ...m,
                    statusText: '',
                    toolCalls: [...(m.toolCalls || []), { type: 'tool_result', name: data.name, result: data.result }]
                  }
                : m
            ))
          } else if (eventType === 'retry') {
            // API retry event - show retry status
            const attempt = data.attempt || 2
            const maxRetries = data.max_retries || 3
            setMessages(prev => prev.map((m, idx) =>
              idx === prev.length - 1 && m.isStreaming
                ? { ...m, statusText: `${t('hyperAi.retrying', 'Retrying')} (${attempt}/${maxRetries})...` }
                : m
            ))
          } else if (eventType === 'interrupted') {
            isInterrupted = true
            interruptedRound = data.round || 0
            if (data.conversation_id) {
              setCurrentConvId(data.conversation_id)
            }
          } else if (eventType === 'error') {
            console.error('Stream error:', data.message)
            break
          } else if (eventType === 'done') {
            if (data.content) content = data.content
            if (data.conversation_id) setCurrentConvId(data.conversation_id)
            if (data.token_usage) setTokenUsage(data.token_usage)
            if (data.compression_points) setCompressionPoints(data.compression_points)
          }
        }

        offset = next_offset

        // Check if task is done
        if (status === 'completed' || status === 'error') {
          break
        }
      }

      // Finalize message - convert streaming toolCalls to stored format
      const toolCallsLog = toolCalls.filter(tc => tc.type === 'tool_call' || tc.type === 'tool_result')
        .reduce((acc: ToolCallLogEntry[], tc) => {
          if (tc.type === 'tool_call' && tc.name) {
            acc.push({ tool: tc.name, args: tc.args || {}, result: '' })
          } else if (tc.type === 'tool_result' && tc.name && acc.length > 0) {
            // Find matching tool call and add result
            const lastCall = acc[acc.length - 1]
            if (lastCall.tool === tc.name) {
              lastCall.result = tc.result || ''
            }
          }
          return acc
        }, [])

      setMessages(prev => prev.map((m, idx) =>
        idx === prev.length - 1 && m.isStreaming
          ? {
              ...m,
              content: content || m.content,
              reasoning_snapshot: reasoning || undefined,
              tool_calls_log: toolCallsLog.length > 0 ? JSON.stringify(toolCallsLog) : undefined,
              isStreaming: false,
              statusText: undefined,
              toolCalls: undefined,
              isInterrupted,
              interruptedRound: isInterrupted ? interruptedRound : undefined,
              is_complete: !isInterrupted
            }
          : m
      ))
      setStreamingContent('')
      setSending(false)
      fetchConversations()
    } catch (e) {
      console.error('Polling error:', e)
      setMessages(prev => prev.map((m, idx) =>
        idx === prev.length - 1 && m.isStreaming
          ? { ...m, isStreaming: false, content: content || t('hyperAi.connectionLost', 'Connection lost') }
          : m
      ))
      setSending(false)
    }
  }

  const handleContinue = () => {
    setInputValue(t('hyperAi.continueMessage', 'Please continue'))
    setTimeout(() => handleSend(), 100)
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex h-full">
      {/* Left: Conversation List */}
      <div className="w-64 border-r flex flex-col">
        <div className="p-3 border-b">
          <Button onClick={handleNewConversation} className="w-full" size="sm">
            <Plus className="w-4 h-4 mr-2" />
            {t('hyperAi.newChat', 'New Chat')}
          </Button>
        </div>
        <ScrollArea className="flex-1">
          <div className="p-2 space-y-1">
            {conversations.map(conv => (
              <button
                key={conv.id}
                onClick={() => setCurrentConvId(conv.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${
                  currentConvId === conv.id
                    ? 'bg-secondary text-secondary-foreground'
                    : 'hover:bg-muted text-muted-foreground'
                }`}
              >
                <div className="flex items-center gap-2">
                  <MessageSquare className="w-4 h-4 flex-shrink-0" />
                  <span className="truncate">{conv.title}</span>
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {conv.message_count} {t('hyperAi.messages', 'messages')}
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>
      </div>

      {/* Center: Chat Area */}
      <div className="flex-1 flex flex-col min-w-0">
        {messages.length === 0 ? (
          <WelcomeMessage nickname={nickname} t={t} />
        ) : (
          <ScrollArea className="flex-1 p-4">
            <div className="space-y-4 max-w-5xl mx-auto">
              {messages.map((msg, idx) => {
                // Check if this message is a compression point
                const compressionPoint = compressionPoints.find(cp => cp.message_id === msg.id)
                return (
                  <div key={idx}>
                    <MessageBubble
                      message={msg}
                      onContinue={msg.isInterrupted && !sending ? handleContinue : undefined}
                      t={t}
                    />
                    {compressionPoint && (
                      <div className="flex items-center gap-3 my-4 text-xs text-muted-foreground">
                        <div className="flex-1 border-t border-dashed border-muted-foreground/30" />
                        <span className="px-2 py-1 bg-muted rounded text-[10px]">
                          {t('hyperAi.compressionPoint', 'Context compressed')}
                        </span>
                        <div className="flex-1 border-t border-dashed border-muted-foreground/30" />
                      </div>
                    )}
                  </div>
                )
              })}
              <div ref={messagesEndRef} />
            </div>
          </ScrollArea>
        )}

        {/* Input Area */}
        <div className="p-4 border-t">
          <div className="max-w-5xl mx-auto flex gap-2 items-end">
            <textarea
              ref={textareaRef}
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={t('hyperAi.inputPlaceholder', 'Type a message...')}
              disabled={sending}
              className="flex-1 min-h-[80px] max-h-[200px] rounded-md border border-input bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:cursor-not-allowed disabled:opacity-50 resize-y"
              rows={3}
            />
            <Button onClick={handleSend} disabled={!inputValue.trim() || sending} className="h-[80px] px-4">
              {sending ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
            </Button>
          </div>
          <div className="flex justify-between items-center mt-2 max-w-5xl mx-auto">
            <p className="text-xs text-muted-foreground">
              {t('common.keyboardHintCtrlEnter', 'Press Ctrl+Enter (Cmd+Enter on Mac) to send')}
            </p>
            {tokenUsage?.show_warning && (
              <p className="text-xs text-amber-500">
                {t('hyperAi.contextWarning', 'Context remaining: {{percent}}% · Compressing soon', { percent: Math.max(0, Math.round((1 - tokenUsage.usage_ratio) * 100)) })}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Right: Config Panel */}
      {showConfig && (
        <div className="w-[500px] border-l p-4 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-medium flex items-center gap-2">
              <Settings className="w-4 h-4" />
              {t('hyperAi.configTitle', 'Hyper AI Config')}
            </h3>
            <Button variant="ghost" size="icon" onClick={() => setShowConfigModal(true)}>
              <Pencil className="w-4 h-4" />
            </Button>
          </div>

          {profile && (
            <div
              className="space-y-3 text-sm cursor-pointer hover:bg-muted/50 rounded-lg p-2 -mx-2 transition-colors"
              onClick={() => setShowConfigModal(true)}
            >
              <div className="flex items-center">
                <span className="text-muted-foreground shrink-0">Provider:</span>
                <span className="ml-2 truncate">{profile.llm_provider || 'Not configured'}</span>
              </div>
              <div className="flex items-center">
                <span className="text-muted-foreground shrink-0">Model:</span>
                <span className="ml-2 truncate">{profile.llm_model || '-'}</span>
              </div>
              {profile.llm_base_url && (
                <div className="flex items-center">
                  <span className="text-muted-foreground shrink-0">Base URL:</span>
                  <span className="ml-2 text-xs truncate">{profile.llm_base_url}</span>
                </div>
              )}
            </div>
          )}

          <div className="pt-4 border-t">
            <h4 className="text-sm font-medium text-muted-foreground mb-2">
              {t('hyperAi.skills', 'Skills')}
            </h4>
            <p className="text-xs text-muted-foreground">
              {t('hyperAi.skillsComingSoon', 'Coming soon...')}
            </p>
          </div>
        </div>
      )}

      {/* LLM Config Modal */}
      <LLMConfigModal
        open={showConfigModal}
        onClose={() => setShowConfigModal(false)}
        providers={providers}
        currentProfile={profile}
        onSaved={fetchProfile}
      />
    </div>
  )
}

// Message bubble component with avatar, markdown, tool calls, and interrupt recovery
// Wrapped with React.memo to prevent re-rendering when parent state (e.g. inputValue)
// changes but message props remain the same. Without memo, every keystroke in the
// textarea triggers re-render of ALL message bubbles (including expensive ReactMarkdown),
// causing noticeable input lag when conversation history is long.
const MessageBubble = memo(function MessageBubble({
  message,
  onContinue,
  t
}: {
  message: Message
  onContinue?: () => void
  t: (key: string, fallback?: string) => string
}) {
  const [showReasoning, setShowReasoning] = useState(false)
  const [showToolCalls, setShowToolCalls] = useState(false)
  const isUser = message.role === 'user'

  // Parse tool calls log from stored messages
  const toolCallsLog: ToolCallLogEntry[] = message.tool_calls_log
    ? (() => { try { return JSON.parse(message.tool_calls_log) } catch { return [] } })()
    : []

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      {/* Avatar */}
      <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
        isUser ? 'bg-primary text-primary-foreground' : 'bg-muted'
      }`}>
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>

      {/* Message content */}
      <div className={`max-w-[80%] rounded-lg px-4 py-3 ${
        isUser
          ? 'bg-primary text-primary-foreground'
          : 'bg-muted min-w-[400px]'
      }`}>
        {/* Status text during streaming */}
        {message.isStreaming && message.statusText && (
          <div className="flex items-center gap-2 text-xs mb-2 text-primary animate-pulse">
            <PacmanLoader className="w-6 h-3" />
            <span>{message.statusText}</span>
          </div>
        )}

        {/* Real-time tool calls during streaming */}
        {message.isStreaming && message.toolCalls && message.toolCalls.length > 0 && (
          <div className="mb-2 text-xs bg-background/50 rounded p-2 max-h-32 overflow-y-auto">
            {message.toolCalls.slice(-5).map((entry, idx) => (
              <div key={idx} className="mb-1 last:mb-0">
                {entry.type === 'tool_call' && (
                  <span className="text-blue-500">→ {entry.name}</span>
                )}
                {entry.type === 'tool_result' && (
                  <span className="text-green-500">← {entry.name}: done</span>
                )}
                {entry.type === 'reasoning' && (
                  <span className="text-gray-500 italic">{(entry.content || '').slice(0, 100)}...</span>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Tool calls log for completed messages - moved above content */}
        {!message.isStreaming && toolCallsLog.length > 0 && (
          <details className="mb-3 text-xs border rounded-md">
            <summary className="px-3 py-2 cursor-pointer bg-muted/50 hover:bg-muted font-medium flex items-center gap-1">
              <Wrench className="w-3 h-3" />
              {t('hyperAi.toolCallsDetail', 'Tool calls')} ({toolCallsLog.length})
            </summary>
            <div className="p-3 space-y-3 max-h-96 overflow-y-auto">
              {toolCallsLog.map((entry, idx) => (
                <div key={idx} className="border-b pb-2 last:border-b-0 last:pb-0">
                  <div className="font-medium text-blue-600 dark:text-blue-400 mb-1">
                    {idx + 1}. {entry.tool}
                  </div>
                  {entry.args && Object.keys(entry.args).length > 0 && (
                    <div className="mb-1 ml-2 text-muted-foreground">
                      {Object.entries(entry.args).map(([key, value]) => (
                        <div key={key}>{key}: {JSON.stringify(value)}</div>
                      ))}
                    </div>
                  )}
                  <div className="ml-2 text-green-600 dark:text-green-400">
                    Result: {entry.result.length > 200 ? entry.result.slice(0, 200) + '...' : entry.result}
                  </div>
                </div>
              ))}
            </div>
          </details>
        )}

        {/* Reasoning snapshot for completed messages - moved above content */}
        {!message.isStreaming && message.reasoning_snapshot && (
          <details className="mb-3 text-xs border rounded-md">
            <summary className="px-3 py-2 cursor-pointer bg-muted/50 hover:bg-muted font-medium">
              {t('hyperAi.reasoningProcess', 'Reasoning process')}
            </summary>
            <div className="p-3 max-h-96 overflow-y-auto">
              <pre className="whitespace-pre-wrap text-muted-foreground">{message.reasoning_snapshot}</pre>
            </div>
          </details>
        )}

        {/* Main content with Markdown */}
        <div className={`text-sm prose prose-sm max-w-none ${
          isUser ? 'prose-invert' : 'dark:prose-invert'
        }`}>
          {message.content ? (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content}
            </ReactMarkdown>
          ) : message.isStreaming ? (
            <span className="text-muted-foreground italic">{t('hyperAi.generating', 'Generating...')}</span>
          ) : null}
        </div>

        {/* Streaming cursor */}
        {message.isStreaming && message.content && (
          <span className="inline-block w-2 h-4 bg-current animate-pulse ml-1" />
        )}

        {/* Interrupted message - continue button */}
        {message.isInterrupted && onContinue && (
          <div className="mt-3 pt-3 border-t border-border/50">
            <div className="flex items-center gap-2 text-xs text-amber-600 dark:text-amber-400 mb-2">
              <AlertCircle className="w-3 h-3" />
              <span>
                {t('hyperAi.interruptedAt', 'Interrupted at round {{round}}').replace('{{round}}', String(message.interruptedRound || '?'))}
              </span>
            </div>
            <Button size="sm" variant="outline" onClick={onContinue} className="text-xs">
              <Play className="w-3 h-3 mr-1" />
              {t('hyperAi.continueButton', 'Continue')}
            </Button>
          </div>
        )}
      </div>
    </div>
  )
})
