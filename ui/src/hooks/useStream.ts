import { useCallback, useRef, useState } from 'react'
import { streamSSE } from '../lib/stream'
import { summarizeUpdate } from '../lib/summarize'
import type { PendingConfirmation, RunStreamRequest, TraceEntry } from '../lib/types'

interface StreamState {
  isStreaming: boolean
  isHITLPaused: boolean
  hitlPayload: PendingConfirmation | null
  activeNode: string | null
  liveReasoning: string | null
}

interface UseStreamResult extends StreamState {
  runStream: (threadId: string, body: RunStreamRequest, onEntry: (e: TraceEntry) => void) => Promise<void>
  cancelStream: () => void
  clearHITL: () => void
}

// Supervisor entries are "routing" decisions (reasoning), not final answers,
// unless the supervisor itself produced an AI message (e.g. unknown intent reply).
function isSupervisorRouting(updates: Record<string, unknown>): boolean {
  const msgs = (updates.messages as unknown[]) || []
  return !msgs.some((m) => (m as Record<string, unknown>)?.type === 'ai')
}

export function useStream(): UseStreamResult {
  const [state, setState] = useState<StreamState>({
    isStreaming: false,
    isHITLPaused: false,
    hitlPayload: null,
    activeNode: null,
    liveReasoning: null,
  })
  const abortRef = useRef<AbortController | null>(null)

  const cancelStream = useCallback(() => {
    abortRef.current?.abort()
  }, [])

  const clearHITL = useCallback(() => {
    setState((s) => ({ ...s, isHITLPaused: false, hitlPayload: null }))
  }, [])

  const runStream = useCallback(
    async (threadId: string, body: RunStreamRequest, onEntry: (e: TraceEntry) => void) => {
      abortRef.current?.abort()
      const controller = new AbortController()
      abortRef.current = controller

      setState({ isStreaming: true, isHITLPaused: false, hitlPayload: null, activeNode: null, liveReasoning: null })

      // Buffer reasoning steps per node until the node's updates event arrives.
      const reasoningBuffer = new Map<string, string>()

      try {
        for await (const event of streamSSE(threadId, body, controller.signal)) {
          if ('error' in event) {
            onEntry({
              role: 'agent',
              agent: 'supervisor',
              content: `❌ **Error:** \`${event.error}\``,
            })
            break
          }

          // Intermediate reasoning step from a specialist agent's internal LLM turn.
          if ('reasoning_step' in event) {
            const { node, reasoning_step } = event
            const current = reasoningBuffer.get(node) ?? ''
            const updated = current ? `${current}\n\n${reasoning_step}` : reasoning_step
            reasoningBuffer.set(node, updated)
            setState((s) => ({ ...s, activeNode: node, liveReasoning: updated }))
            continue
          }

          if (!('updates' in event)) continue

          const node = event.node

          if (node === '__pause__') {
            const pc = (event.updates as { pending_confirmation: PendingConfirmation }).pending_confirmation
            setState((s) => ({ ...s, isHITLPaused: true, hitlPayload: pc }))
            continue
          }

          if (node === 'memory_loader' || node === 'memory_saver') continue

          const reasoning = reasoningBuffer.get(node) ?? ''
          reasoningBuffer.delete(node)
          // Clear live reasoning display once this node's final update arrives.
          setState((s) => ({ ...s, activeNode: null, liveReasoning: null }))

          const content = summarizeUpdate(node, event.updates)
          const isReasoning = node === 'supervisor' && isSupervisorRouting(event.updates)

          onEntry({
            role: 'agent',
            agent: node,
            content,
            isReasoning,
            reasoning: reasoning || undefined,
          })
        }
      } catch (err) {
        if (err instanceof Error && err.name !== 'AbortError') {
          onEntry({
            role: 'agent',
            agent: 'supervisor',
            content: `❌ **Error:** \`${err.message}\``,
          })
        }
      } finally {
        setState((s) => ({ ...s, isStreaming: false, activeNode: null, liveReasoning: null }))
      }
    },
    [],
  )

  return { ...state, runStream, cancelStream, clearHITL }
}
