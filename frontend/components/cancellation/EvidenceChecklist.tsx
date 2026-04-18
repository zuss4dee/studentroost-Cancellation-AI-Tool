type EvidenceChecklistProps = {
  items: string[];
  title?: string;
 className?: string;
};

export function EvidenceChecklist({ items, title = "Evidence to upload", className = "" }: EvidenceChecklistProps) {
  if (items.length === 0) return null;
  return (
    <div className={`rounded-xl border border-[#DEE2E6] bg-[#F8F9FA] px-3 py-3 ${className}`}>
      <div className="mb-2 text-[12px] font-semibold text-gray-800">{title}</div>
      <ul className="list-inside list-disc space-y-1.5 text-[13px] leading-relaxed text-gray-800">
        {items.map((line, i) => (
          <li key={i}>{line}</li>
        ))}
      </ul>
    </div>
  );
}
