import type { Session, TraceEntry } from './types'

const SESSIONS_KEY = 'smartfin_sessions'
const TRACE_PREFIX = 'smartfin_trace_'

export function generateThreadId(): string {
  return `ui-${crypto.randomUUID().replace(/-/g, '').slice(0, 10)}`
}

export function autoTitle(firstMessage: string): string {
  return firstMessage.trim().slice(0, 40) || 'New chat'
}

export function listSessions(): Session[] {
  try {
    const raw = localStorage.getItem(SESSIONS_KEY)
    const sessions: Session[] = raw ? JSON.parse(raw) : []
    return sessions.sort((a, b) => b.lastActivityAt - a.lastActivityAt)
  } catch {
    return []
  }
}

export function getSession(threadId: string): Session | undefined {
  return listSessions().find((s) => s.threadId === threadId)
}

export function createSession(title = 'New chat'): Session {
  const session: Session = {
    threadId: generateThreadId(),
    title,
    createdAt: Date.now(),
    lastActivityAt: Date.now(),
  }
  const sessions = listSessions()
  sessions.unshift(session)
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
  return session
}

export function renameSession(threadId: string, title: string): void {
  const sessions = listSessions()
  const idx = sessions.findIndex((s) => s.threadId === threadId)
  if (idx !== -1) {
    sessions[idx].title = title
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
  }
}

export function touchSession(threadId: string): void {
  const sessions = listSessions()
  const idx = sessions.findIndex((s) => s.threadId === threadId)
  if (idx !== -1) {
    sessions[idx].lastActivityAt = Date.now()
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
  }
}

export function deleteSession(threadId: string): void {
  const sessions = listSessions().filter((s) => s.threadId !== threadId)
  localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions))
  localStorage.removeItem(`${TRACE_PREFIX}${threadId}`)
}

export function loadTrace(threadId: string): TraceEntry[] {
  try {
    const raw = localStorage.getItem(`${TRACE_PREFIX}${threadId}`)
    return raw ? JSON.parse(raw) : []
  } catch {
    return []
  }
}

export function saveTrace(threadId: string, trace: TraceEntry[]): void {
  localStorage.setItem(`${TRACE_PREFIX}${threadId}`, JSON.stringify(trace))
}
