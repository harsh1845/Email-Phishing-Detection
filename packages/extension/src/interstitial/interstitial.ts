const params = new URLSearchParams(location.search);
const url = params.get('url') || '';
const reasons = (params.get('reasons') || '').split('\n').filter(Boolean);

document.querySelector('#url')!.textContent = url;
const list = document.querySelector('#reasons')!;
for (const reason of reasons) {
  const li = document.createElement('li');
  li.textContent = reason;
  list.appendChild(li);
}

document.querySelector('#back')!.addEventListener('click', () => {
  history.length > 1 ? history.back() : window.close();
});

document.querySelector('#proceed')!.addEventListener('click', () => {
  if (url) location.href = url;
});
