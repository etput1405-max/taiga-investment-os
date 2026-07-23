from __future__ import annotations
import io,json,math,os,re,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np,pandas as pd,requests,yfinance as yf

JST=timezone(timedelta(hours=9)); OUTPUT=Path("data.json")
MAX_STOCKS=int(os.getenv("MAX_STOCKS","800")); BATCH_SIZE=int(os.getenv("BATCH_SIZE","80"))
MARKETS={"^N225":"日経平均","1306.T":"TOPIX連動ETF","JPY=X":"ドル円","^TNX":"米国10年金利","^VIX":"VIX","CL=F":"WTI原油","GC=F":"金","HG=F":"銅"}
FALLBACK={"1605.T":"INPEX","1928.T":"積水ハウス","2914.T":"JT","4063.T":"信越化学","4502.T":"武田薬品","4519.T":"中外製薬","4568.T":"第一三共","6098.T":"リクルート","6301.T":"コマツ","6501.T":"日立","6758.T":"ソニーG","6861.T":"キーエンス","6902.T":"デンソー","7203.T":"トヨタ","7267.T":"ホンダ","7741.T":"HOYA","7974.T":"任天堂","8001.T":"伊藤忠","8031.T":"三井物産","8035.T":"東京エレクトロン","8058.T":"三菱商事","8306.T":"三菱UFJ","8316.T":"三井住友FG","8411.T":"みずほFG","8766.T":"東京海上","9432.T":"NTT","9433.T":"KDDI","9983.T":"ファーストリテイリング","9984.T":"ソフトバンクG"}
JPX_PAGE="https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"

def universe():
    try:
        h=requests.get(JPX_PAGE,timeout=30,headers={"User-Agent":"Mozilla/5.0"}).text
        links=re.findall(r'href="([^"]+\.(?:xls|xlsx))"',h,re.I)
        for u in links:
            if u.startswith("/"):u="https://www.jpx.co.jp"+u
            try:
                b=requests.get(u,timeout=45,headers={"User-Agent":"Mozilla/5.0"}).content; d=pd.read_excel(io.BytesIO(b))
                cc=next(c for c in d.columns if "コード" in str(c) or "Code" in str(c)); nc=next(c for c in d.columns if "銘柄名" in str(c) or "Issue name" in str(c))
                mc=next((c for c in d.columns if "市場・商品区分" in str(c) or "Market" in str(c)),None)
                if mc is not None:d=d[d[mc].astype(str).str.contains("プライム|スタンダード|グロース|Prime|Standard|Growth",regex=True)]
                out={}
                for _,r in d.iterrows():
                    code=str(r[cc]).split(".")[0].strip()
                    if re.fullmatch(r"\d{4}",code):out[code+".T"]=str(r[nc]).strip()
                if len(out)>1000:return out
            except Exception:pass
    except Exception:pass
    return FALLBACK.copy()

def pct(s,n):
    s=s.dropna()
    return None if len(s)<=n or not s.iloc[-n-1] else float((s.iloc[-1]/s.iloc[-n-1]-1)*100)
def z(s):
    sd=s.std(ddof=0); return (s-s.mean())/sd if sd and not np.isnan(sd) else pd.Series(0.0,index=s.index)
def dl(symbols):
    fs=[]
    for i in range(0,len(symbols),BATCH_SIZE):
        try:
            d=yf.download(symbols[i:i+BATCH_SIZE],period="1y",auto_adjust=True,progress=False,threads=True,group_by="column")
            if not d.empty:fs.append(d["Close"] if isinstance(d.columns,pd.MultiIndex) else d[["Close"]].rename(columns={"Close":symbols[i]}))
        except Exception:pass
        time.sleep(1)
    return pd.concat(fs,axis=1).loc[:,lambda x:~x.columns.duplicated()].sort_index().ffill() if fs else pd.DataFrame()

def main():
    u=universe(); selected=dict(list(sorted(u.items()))[:MAX_STOCKS]); c=dl(list(selected)); m=dl(list(MARKETS))
    market=[]
    for s,n in MARKETS.items():
        if s in m.columns and not m[s].dropna().empty:
            x=m[s].dropna(); market.append({"symbol":s,"name":n,"value":round(float(x.iloc[-1]),4),"change20":round(pct(x,20) or 0,2)})
    mm={x["symbol"]:x for x in market}; nk=mm.get("^N225",{}).get("change20",0); v=mm.get("^VIX",{}).get("change20",0); cu=mm.get("HG=F",{}).get("change20",0)
    rs=float(np.clip(50+nk*.8-max(v,0)*.35+cu*.25,0,100)); regime={"score":round(rs,1),"label":"リスクオン" if rs>=65 else "リスクオフ" if rs<=35 else "中立"}
    raw=[]
    for t,n in selected.items():
        if t not in c.columns:continue
        s=c[t].dropna()
        if len(s)<130:continue
        r=s.pct_change().dropna(); p=float(s.iloc[-1]); h=float(s.tail(252).max())
        raw.append({"ticker":t,"name":n,"price":p,"mom20":pct(s,20),"mom60":pct(s,60),"mom120":pct(s,120),"drawdown":(p/h-1)*100,"vol20":float(r.tail(20).std(ddof=0)*math.sqrt(252)*100)})
    if not raw:raise RuntimeError("株価データ取得失敗")
    d=pd.DataFrame(raw).set_index("ticker"); f=pd.DataFrame(index=d.index)
    f["momentum"]=z(d.mom20.fillna(0))*.35+z(d.mom60.fillna(0))*.40+z(d.mom120.fillna(0))*.25; f["pullback"]=z(-abs(d.drawdown.fillna(0)+12)); f["stability"]=z(-d.vol20.fillna(d.vol20.median())); f["relative"]=z(d.mom60.fillna(0)-nk)
    comp=f.momentum*.40+f.pullback*.25+f.stability*.15+f.relative*.20
    d["score"]=(comp.rank(pct=True)*100).round(1); d["momentum_score"]=(f.momentum.rank(pct=True)*100).round(1); d["pullback_score"]=(f.pullback.rank(pct=True)*100).round(1); d["stability_score"]=(f.stability.rank(pct=True)*100).round(1)
    med=float(d.vol20.median()); ranking=[]
    for t,r in d.sort_values("score",ascending=False).iterrows():
        sc=float(r.score); sig="最優先候補" if sc>=85 else "買い場候補" if sc>=70 else "監視" if sc>=50 else "見送り"; why=[]
        if (r.mom20 or 0)>0 and (r.mom60 or 0)>0:why.append("短中期上向き")
        if -20<=r.drawdown<=-5:why.append("適度な押し目")
        if (r.mom60 or 0)>nk:why.append("日経平均より強い")
        if r.vol20<med:why.append("値動き比較的安定")
        ranking.append({"ticker":t,"name":r["name"],"price":round(float(r.price),2),"score":sc,"signal":sig,"mom20":round(float(r.mom20),2) if pd.notna(r.mom20) else None,"mom60":round(float(r.mom60),2) if pd.notna(r.mom60) else None,"mom120":round(float(r.mom120),2) if pd.notna(r.mom120) else None,"drawdown":round(float(r.drawdown),2),"vol20":round(float(r.vol20),2),"momentum_score":float(r.momentum_score),"pullback_score":float(r.pullback_score),"stability_score":float(r.stability_score),"reasons":why or ["相対順位で抽出"]})
    OUTPUT.write_text(json.dumps({"updated_at":datetime.now(JST).isoformat(timespec="seconds"),"regime":regime,"market":market,"ranking":ranking,"meta":{"universe":len(u),"requested":len(selected),"success":len(raw),"failed":len(selected)-len(raw)}},ensure_ascii=False,indent=2),encoding="utf-8")
if __name__=="__main__":main()
