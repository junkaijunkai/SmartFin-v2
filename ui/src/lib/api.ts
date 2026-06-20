import type { ThreadState } from './types'

export async function getThreadState(threadId: string): Promise<ThreadState> {
  const res = await fetch(`/threads/${threadId}/state`)
  if (!res.ok) throw new Error(`Failed to get thread state: ${res.status}`)
  return res.json()
}

export async function deleteThread(threadId: string): Promise<void> {
  await fetch(`/threads/${threadId}`, { method: 'DELETE' })
}
