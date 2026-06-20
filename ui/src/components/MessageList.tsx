import { useEffect, useRef } from 'react'
import type { TraceEntry } from '../lib/types'
import { MessageBubble, AGENT_META } from './MessageBubble'

interface MessageListProps {
  trace: TraceEntry[]
  isStreaming?: boolean
  activeNode?: string | null
  liveReasoning?: string | null
  onEditSave?: (index: number, newText: string) => void
  hitlCard?: React.ReactNode
}

export function MessageList({ trace, isStreaming, activeNode, liveReasoning, onEditSave, hitlCard }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [trace.length, isStreaming, liveReasoning, hitlCard])

  const [activeEmoji, activeName] = AGENT_META[activeNode ?? ''] ?? ['🤖', activeNode ?? 'Agent']

  return (
    <div className="flex-1 overflow-y-auto px-4 pt-6 pb-6">
      <div className="max-w-3xl mx-auto">
        {trace.length === 0 && !isStreaming && (
          <div className="flex flex-col items-center justify-center h-full min-h-[300px] text-center gap-3">
            <span className="text-4xl">💬</span>
            <p className="text-gray-500 text-sm">开始一个新对话，SmartFin 会帮助你管理财务</p>
          </div>
        )}
        {trace.map((entry, i) => (
          <MessageBubble key={i} entry={entry} index={i} onEditSave={onEditSave} />
        ))}

        {/* Live reasoning card: shown while an agent's ReAct loop is in progress */}
        {isStreaming && activeNode && liveReasoning && (
          <div className="mb-4 border border-gray-200 rounded-xl overflow-hidden shadow-sm">
            {/* Card header with "Thinking…" label */}
            <div className="flex items-center gap-2 px-3 py-2 bg-gray-50 border-b border-gray-100">
              <span className="text-[11px] font-semibold text-gray-600 tracking-wide">Thinking…</span>
              <div className="flex gap-0.5">
                {[0, 1, 2].map((n) => (
                  <div
                    key={n}
                    className="w-1 h-1 rounded-full bg-gray-400 animate-bounce"
                    style={{ animationDelay: `${n * 0.15}s` }}
                  />
                ))}
              </div>
              <span className="ml-auto text-[11px] text-gray-400">
                {activeEmoji} {activeName}
              </span>
            </div>
            {/* Reasoning steps */}
            <div className="px-3 py-2.5 space-y-2.5 bg-white">
              {liveReasoning.split('\n\n').filter((s) => s.trim()).map((step, i) => (
                <div key={i} className="flex gap-2.5">
                  <span className="text-[10px] font-mono text-gray-300 mt-0.5 flex-shrink-0 w-5 text-right">
                    {i + 1}
                  </span>
                  <p className="text-xs text-gray-500 leading-relaxed whitespace-pre-wrap">{step.trim()}</p>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Generic processing indicator when no reasoning is available yet */}
        {isStreaming && (!activeNode || !liveReasoning) && (
          <div className="flex items-center gap-2 mb-4 text-gray-400">
            <div className="flex gap-1">
              {[0, 1, 2].map((n) => (
                <div
                  key={n}
                  className="w-1.5 h-1.5 rounded-full bg-gray-400 animate-bounce"
                  style={{ animationDelay: `${n * 0.15}s` }}
                />
              ))}
            </div>
            <span className="text-xs">Processing…</span>
          </div>
        )}

        {/* HITL confirmation card — rendered inside the scroll area so it doesn't overlap ChatInput */}
        {hitlCard && (
          <div className="mb-4">
            {hitlCard}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  )
}
