import type { RunStreamRequest, SSEEvent } from './types'

export async function* streamSSE(
  threadId: string,
  body: RunStreamRequest,
  signal?: AbortSignal,
): AsyncGenerator<SSEEvent> {
  const response = await fetch(`/threads/${threadId}/runs/stream`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    throw new Error(`Stream request failed: ${response.status} ${response.statusText}`)
  }

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue
        const data = line.slice(6).trim()
        if (data === '[DONE]') return
        try {
          yield JSON.parse(data) as SSEEvent
        } catch {
          // skip malformed JSON
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
}
