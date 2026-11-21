/* Shared Google account actions for the candidate and recruiter portals. */
window.jobScopeGoogle = {
  async start(intent) {
    const response = await fetch('/api/google/start', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ intent }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || 'Could not connect to Google.');
    window.location.assign(result.url);
  },
  notice() {
    const url = new URL(window.location.href);
    const code = url.searchParams.get('google');
    if (!code) return '';
    url.searchParams.delete('google');
    history.replaceState(null, '', url.pathname + url.search + url.hash);
    return {
      linked: 'Google sign-in is linked to your JobScope account.',
      gmail_connected: 'Gmail is connected. Select Gmail to fetch your job alerts.',
      link_first: 'Sign in with your existing JobScope account first, then select Link Google sign-in.',
      cancelled: 'The Google connection was cancelled.',
      expired: 'The Google request expired. Start it again from JobScope.',
      failed: 'Google could not complete the connection. Check the account you selected and try again.',
    }[code] || '';
  },
};
