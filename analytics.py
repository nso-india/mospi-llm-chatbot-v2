"""
Analytics and Monitoring Service
Handles user tracking, performance metrics, and report generation
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import logging
from models import RequestMetrics, UserActivity, Interaction, MonthlyReport, find_interactions_safe
from collections import defaultdict

logger = logging.getLogger("chatbot")
from beanie.operators import In

# IST timezone (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

def ist_now() -> datetime:
    """Get current datetime in IST timezone"""
    return datetime.now(IST)

def normalize_to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalize a datetime to IST-aware. Handles both naive and aware datetimes."""
    if dt is None:
        return None
    # If timezone-naive, assume it's IST (for backward compatibility with old records)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=IST)
    # If timezone-aware, convert to IST
    return dt.astimezone(IST)


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track response times for all API endpoints"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip static files and non-API routes
        if request.url.path.startswith("/static/") or request.url.path == "/":
            return await call_next(request)
        
        start_time = time.time()
        endpoint = request.url.path
        method = request.method
        
        try:
            response = await call_next(request)
            status_code = response.status_code
            error = None
        except Exception as e:
            status_code = 500
            error = str(e)
            raise
        finally:
            response_time_ms = (time.time() - start_time) * 1000
            
            # Log metrics asynchronously (don't block request)
            try:
                metrics = RequestMetrics(
                    endpoint=endpoint,
                    method=method,
                    response_time_ms=response_time_ms,
                    status_code=status_code,
                    error=error,
                    timestamp=ist_now()
                )
                await metrics.insert()
            except Exception as e:
                logger.error(f"Failed to save request metrics: {e}", exc_info=True)
        
        return response


class AnalyticsService:
    """Service for analytics and reporting"""
    
    @staticmethod
    async def track_user_activity(
        session_id: str,
        device_id: Optional[str] = None,
        source: Optional[str] = None,
        from_proxy: bool = False,
    ):
        """Track or update user activity. Only proxy sessions (from_proxy=True) are stored and counted."""
        try:
            from session_store import get_session_track
            track = get_session_track(session_id)
            if track is False:
                return  # Session from our React app — do not track
            activity = await UserActivity.find_one(UserActivity.session_id == session_id)
            
            if activity:
                activity.last_seen = ist_now()
                activity.total_interactions += 1
                activity.is_active = True
                if device_id and not activity.device_id:
                    activity.device_id = device_id
                if source and not activity.source:
                    activity.source = source
                if from_proxy:
                    activity.from_proxy = True
                await activity.save()
            else:
                activity = UserActivity(
                    session_id=session_id,
                    device_id=device_id,
                    source=source,
                    from_proxy=from_proxy,
                    first_seen=ist_now(),
                    last_seen=ist_now(),
                    total_interactions=1,
                    total_sessions=1,
                    is_active=True
                )
                await activity.insert()
                if from_proxy:
                    logger.info(f"user_activity: inserted session_id={session_id[:8]}... from_proxy=True source={source or 'n/a'}")
        except Exception as e:
            logger.error(f"Failed to track user activity: {e}", exc_info=True)
    
    @staticmethod
    async def get_user_statistics(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get user statistics, optionally filtered by date range"""
        try:
            # Only count rows that were created from proxy (from_proxy=True)
            all_users = await UserActivity.find(UserActivity.from_proxy == True).to_list()
            
            # Filter users by first_seen if date range provided
            if start_date or end_date:
                filtered_users = []
                for u in all_users:
                    # Normalize timestamps to IST-aware for comparison
                    first_seen_ist = normalize_to_ist(u.first_seen)
                    if start_date and first_seen_ist < start_date:
                        continue
                    if end_date and first_seen_ist > end_date:
                        continue
                    filtered_users.append(u)
                total_users = len(filtered_users)
            else:
                total_users = len(all_users)
            
            # Get interactions filtered by date to calculate accurate counts
            all_interactions = await find_interactions_safe()
            if start_date or end_date:
                filtered_interactions = []
                for i in all_interactions:
                    # Normalize timestamp to IST-aware for comparison
                    timestamp_ist = normalize_to_ist(i.timestamp)
                    if start_date and timestamp_ist < start_date:
                        continue
                    if end_date and timestamp_ist > end_date:
                        continue
                    filtered_interactions.append(i)
                
                
                total_interactions = len(filtered_interactions)
                
                # Get unique session IDs from interactions
                active_session_ids = set(i.session_id for i in filtered_interactions)
                
                # Filter these sessions to only include those from proxy
                # We need to find which of these session_ids exist in UserActivity with from_proxy=True
                valid_proxy_sessions = await UserActivity.find(
                    In(UserActivity.session_id, list(active_session_ids)),
                    UserActivity.from_proxy == True
                ).to_list()
                
                # Count distinct legitimate sessions
                total_users = len(valid_proxy_sessions)
                
                # "Active Users" is now synonymous with Total Users in a time window
                active_users = total_users
            else:
                # Default view (Lifetime): Total unique users ever seen (already filtered by from_proxy in line 111)
                total_users = len(all_users)
                # Count interactions only for proxy users
                proxy_session_ids = [u.session_id for u in all_users]
                total_interactions = await Interaction.find(
                    Interaction.session_id.in_(proxy_session_ids)
                ).count()
                
                thirty_days_ago = ist_now() - timedelta(days=30)
                active_users = len([u for u in all_users if normalize_to_ist(u.last_seen) >= thirty_days_ago])
            
            avg_interactions = round(total_interactions / total_users, 2) if total_users > 0 else 0
            
            return {
                "total_users": total_users,
                "active_users_30d": active_users,
                "total_interactions": total_interactions,
                "avg_interactions_per_user": avg_interactions
            }
        except Exception as e:
            logger.error(f"Failed to get user statistics: {e}", exc_info=True)
            return {}
    
    @staticmethod
    async def get_performance_metrics(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get performance metrics"""
        try:
            query = {}
            if start_date:
                query["timestamp"] = {"$gte": start_date}
            if end_date:
                if "timestamp" in query:
                    query["timestamp"]["$lte"] = end_date
                else:
                    query["timestamp"] = {"$lte": end_date}
            
            # Get all request metrics
            all_metrics = await RequestMetrics.find_all().to_list()
            
            if not all_metrics:
                return {
                    "total_requests": 0,
                    "avg_response_time_ms": 0,
                    "p50_response_time_ms": 0,
                    "p95_response_time_ms": 0,
                    "p99_response_time_ms": 0,
                    "error_rate": 0,
                    "total_errors": 0
                }
            
            # Filter by date if provided
            if start_date or end_date:
                filtered_metrics = []
                for m in all_metrics:
                    # Normalize timestamp to IST-aware for comparison
                    timestamp_ist = normalize_to_ist(m.timestamp)
                    if start_date and timestamp_ist < start_date:
                        continue
                    if end_date and timestamp_ist > end_date:
                        continue
                    filtered_metrics.append(m)
                all_metrics = filtered_metrics
            
            response_times = sorted([m.response_time_ms for m in all_metrics])
            total_requests = len(response_times)
            
            if total_requests == 0:
                return {"total_requests": 0}
            
            # Calculate percentiles
            p50_idx = int(total_requests * 0.5)
            p95_idx = int(total_requests * 0.95)
            p99_idx = int(total_requests * 0.99)
            
            avg_response_time = sum(response_times) / total_requests
            p50 = response_times[p50_idx] if p50_idx < total_requests else response_times[-1]
            p95 = response_times[p95_idx] if p95_idx < total_requests else response_times[-1]
            p99 = response_times[p99_idx] if p99_idx < total_requests else response_times[-1]
            
            # Error rate
            errors = [m for m in all_metrics if m.status_code >= 400 or m.error]
            error_rate = (len(errors) / total_requests) * 100 if total_requests > 0 else 0
            
            # Get endpoint-specific metrics
            endpoint_stats = defaultdict(lambda: {"count": 0, "total_time": 0, "errors": 0})
            for m in all_metrics:
                endpoint_stats[m.endpoint]["count"] += 1
                endpoint_stats[m.endpoint]["total_time"] += m.response_time_ms
                if m.status_code >= 400 or m.error:
                    endpoint_stats[m.endpoint]["errors"] += 1
            
            endpoint_metrics = []
            for endpoint, stats in endpoint_stats.items():
                endpoint_metrics.append({
                    "endpoint": endpoint,
                    "request_count": stats["count"],
                    "avg_response_time_ms": round(stats["total_time"] / stats["count"], 2),
                    "error_count": stats["errors"],
                    "error_rate": round((stats["errors"] / stats["count"]) * 100, 2) if stats["count"] > 0 else 0
                })
            
            return {
                "total_requests": total_requests,
                "avg_response_time_ms": round(avg_response_time, 2),
                "p50_response_time_ms": round(p50, 2),
                "p95_response_time_ms": round(p95, 2),
                "p99_response_time_ms": round(p99, 2),
                "error_rate": round(error_rate, 2),
                "total_errors": len(errors),
                "endpoint_metrics": sorted(endpoint_metrics, key=lambda x: x["request_count"], reverse=True)
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}", exc_info=True)
            return {}
    
    @staticmethod
    async def get_engagement_metrics(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> Dict:
        """Get engagement metrics"""
        try:
            # Get all interactions
            all_interactions = await find_interactions_safe()
            
            # Filter by date if provided
            if start_date or end_date:
                filtered = []
                for i in all_interactions:
                    # Normalize timestamp to IST-aware for comparison
                    timestamp_ist = normalize_to_ist(i.timestamp)
                    if start_date and timestamp_ist < start_date:
                        continue
                    if end_date and timestamp_ist > end_date:
                        continue
                    filtered.append(i)
                all_interactions = filtered
            
            total_interactions = len(all_interactions)
            
            # Group by session
            sessions = defaultdict(lambda: {"interactions": [], "start_time": None, "end_time": None})
            for i in all_interactions:
                sessions[i.session_id]["interactions"].append(i)
                # Normalize timestamps for comparison
                timestamp_ist = normalize_to_ist(i.timestamp)
                start_time_ist = normalize_to_ist(sessions[i.session_id]["start_time"])
                end_time_ist = normalize_to_ist(sessions[i.session_id]["end_time"])
                
                if not sessions[i.session_id]["start_time"] or timestamp_ist < start_time_ist:
                    sessions[i.session_id]["start_time"] = timestamp_ist
                if not sessions[i.session_id]["end_time"] or timestamp_ist > end_time_ist:
                    sessions[i.session_id]["end_time"] = timestamp_ist
            
            total_sessions = len(sessions)
            
            # Calculate averages
            avg_interactions_per_session = total_interactions / total_sessions if total_sessions > 0 else 0
            
            # Session durations
            durations = []
            for session_id, data in sessions.items():
                if data["start_time"] and data["end_time"]:
                    duration = (data["end_time"] - data["start_time"]).total_seconds() / 60  # minutes
                    durations.append(duration)
            
            avg_session_duration = sum(durations) / len(durations) if durations else 0
            
            # Top queries
            query_counts = defaultdict(int)
            for i in all_interactions:
                query_counts[i.query] += 1
            
            top_queries = sorted(
                [{"query": q, "count": c} for q, c in query_counts.items()],
                key=lambda x: x["count"],
                reverse=True
            )[:10]
            
            # Fallback rate
            fallback_messages = [
                "This seems to be outside my scope. Unfortunately, I am unable to assist you with your requested query. Thank you for your understanding.",
                "यह मेरे दायरे से बाहर लगता है। दुर्भाग्य से, मैं आपके अनुरोधित प्रश्न में सहायता नहीं कर सकता। धन्यवाद।"
            ]
            fallback_count = len([i for i in all_interactions if i.response.strip() in fallback_messages])
            fallback_rate = (fallback_count / total_interactions * 100) if total_interactions > 0 else 0
            
            # Feedback stats
            likes = len([i for i in all_interactions if i.feedback == "like"])
            dislikes = len([i for i in all_interactions if i.feedback == "dislike"])
            total_feedback = likes + dislikes
            feedback_rate = (total_feedback / total_interactions * 100) if total_interactions > 0 else 0
            
            # Peak usage hours: all 24 hours in chronological order (0-23), count of interactions per hour
            hour_counts = defaultdict(int)
            for i in all_interactions:
                # Normalize to IST to get correct hour
                timestamp_ist = normalize_to_ist(i.timestamp)
                hour_counts[timestamp_ist.hour] += 1
            peak_hours = [{"hour": h, "interactions": hour_counts.get(h, 0)} for h in range(24)]
            
            # Interaction Trend (Dynamic)
            trend_data = [] # [{"label": "10:00", "value": 5}, ...]
            
            # Determine grouping strategy based on duration
            duration_days = 30
            if start_date and end_date:
                duration_days = (end_date - start_date).days
            elif start_date:
                duration_days = (ist_now() - start_date).days
            
            # Use 'Last 24 Hours' logic if duration is <= 1 day, or if it's the specific 24h filter
            is_24h_view = duration_days <= 1
            
            if is_24h_view:
                # Create slots from start_date to end_date (all dates already in IST)
                # Base range on the actual filter limits if they exist, else default to 24h
                t_end_ist = end_date if end_date else ist_now()
                t_start_ist = start_date if start_date else (t_end_ist - timedelta(hours=24))
                
                # Round down start to the nearest hour to ensure we catch the first partial hour
                current_hour = t_start_ist.replace(minute=0, second=0, microsecond=0)
                
                slots = {}
                # Create hourly slots until we pass the end time
                while current_hour <= t_end_ist:
                    key = current_hour.strftime("%Y-%m-%d-%H")
                    label = current_hour.strftime("%H:00")
                    slots[key] = {"label": label, "value": 0, "sort_key": current_hour}
                    # Move to next hour
                    current_hour += timedelta(hours=1)
                
                trend_data = [] # Reset to populate from slots later
                # We'll populate trend_data after filling values

                
                # Fill data
                for i in all_interactions:
                    # Normalize timestamp to IST-aware
                    t_ist = normalize_to_ist(i.timestamp)
                    key = t_ist.strftime("%Y-%m-%d-%H")
                    if key in slots:
                        slots[key]["value"] += 1
                
                # Convert slots dict to sorted list for the chart
                trend_data = sorted(slots.values(), key=lambda x: x["sort_key"])
            
            elif duration_days <= 7:
                # Group by Day Name (Mon, Tue...) for last 7 days (all dates already in IST)
                # Initialize last 7 days
                now_ist = ist_now()
                if end_date:
                    now_ist = end_date
                
                slots = {}
                for d in range(7):
                    t = now_ist - timedelta(days=6-d)
                    key = t.strftime("%Y-%m-%d")
                    label = t.strftime("%a") # Mon
                    slots[key] = {"label": label, "value": 0, "sort_key": t}
                    trend_data.append(slots[key])
                
                for i in all_interactions:
                    # Normalize timestamp to IST-aware
                    t_ist = normalize_to_ist(i.timestamp)
                    key = t_ist.strftime("%Y-%m-%d")
                    if key in slots:
                        slots[key]["value"] += 1
                        
            else:
                # Group by Date (MM-DD) for larger ranges
                # Filter is already applied to all_interactions, so just group them
                date_counts = defaultdict(int)
                for i in all_interactions:
                    # Normalize timestamp to IST-aware
                    t_ist = normalize_to_ist(i.timestamp)
                    key = t_ist.strftime("%Y-%m-%d")
                    date_counts[key] += 1
                
                # Sort by date
                sorted_dates = sorted(date_counts.keys())
                
                # If we want a continuous line, we might want to fill gaps, 
                # but for simplicity let's just show present days if gap is huge.
                # However, for charts, filling gaps is better. 
                
                if start_date and end_date:
                    curr = start_date
                    while curr <= end_date:
                        key = curr.strftime("%Y-%m-%d")
                        label = curr.strftime("%b %d")
                        val = date_counts.get(key, 0)
                        trend_data.append({"label": label, "value": val})
                        curr += timedelta(days=1)
                else:
                    # Fallback if no specific range, just show what we have
                    for date_str in sorted_dates:
                        dt = datetime.strptime(date_str, "%Y-%m-%d")
                        trend_data.append({
                            "label": dt.strftime("%b %d"),
                            "value": date_counts[date_str]
                        })
            
            
            return {
                "total_interactions": total_interactions,
                "total_sessions": total_sessions,
                "avg_interactions_per_session": round(avg_interactions_per_session, 2),
                "avg_session_duration_minutes": round(avg_session_duration, 2),
                "fallback_rate": round(fallback_rate, 2),
                "fallback_count": fallback_count,
                "feedback_stats": {
                    "likes": likes,
                    "dislikes": dislikes,
                    "total_feedback": total_feedback,
                    "feedback_rate": round(feedback_rate, 2),
                    "like_percentage": round((likes / total_feedback * 100), 2) if total_feedback > 0 else 0
                },
                "top_queries": top_queries,
                "peak_usage_hours": peak_hours,
                "interaction_trend": trend_data
            }
        except Exception as e:
            logger.error(f"Failed to get engagement metrics: {e}", exc_info=True)
            return {}
    
    @staticmethod
    async def generate_monthly_report(month: Optional[str] = None) -> Dict:
        """Generate monthly analytics report"""
        try:
            # Determine month
            if month:
                year, month_num = month.split("-")
                report_start = datetime(int(year), int(month_num), 1, tzinfo=IST)
            else:
                # Current month (IST): month-to-date = 1st through end of today
                now = ist_now()
                report_start = datetime(now.year, now.month, 1, tzinfo=IST)
                report_end = datetime(now.year, now.month, now.day, tzinfo=IST) + timedelta(days=1)
            
            if month:
                # Explicit month = full calendar month
                if report_start.month == 12:
                    report_end = datetime(report_start.year + 1, 1, 1, tzinfo=IST)
                else:
                    report_end = datetime(report_start.year, report_start.month + 1, 1, tzinfo=IST)
            
            report_month_str = report_start.strftime("%Y-%m")
            
            # Check if report already exists
            existing_report = await MonthlyReport.find_one(MonthlyReport.report_month == report_month_str)
            if existing_report:
                logger.info(f"Report for {report_month_str} already exists, regenerating...")
                await existing_report.delete()
            
            # Get all metrics for the month
            user_stats = await AnalyticsService.get_user_statistics(report_start, report_end)
            performance_stats = await AnalyticsService.get_performance_metrics(report_start, report_end)
            engagement_stats = await AnalyticsService.get_engagement_metrics(report_start, report_end)
            
            # Get interactions for the month
            interactions = await find_interactions_safe()
            month_interactions = [
                i for i in interactions
                if report_start <= normalize_to_ist(i.timestamp) < report_end
            ]
            
            # Calculate response time metrics from interactions
            response_times = [i.response_time_ms for i in month_interactions if i.response_time_ms]
            if response_times:
                response_times_sorted = sorted(response_times)
                avg_response_time = sum(response_times) / len(response_times)
                p95_idx = int(len(response_times_sorted) * 0.95)
                p99_idx = int(len(response_times_sorted) * 0.99)
                p95_response_time = response_times_sorted[p95_idx] if p95_idx < len(response_times_sorted) else response_times_sorted[-1]
                p99_response_time = response_times_sorted[p99_idx] if p99_idx < len(response_times_sorted) else response_times_sorted[-1]
            else:
                avg_response_time = 0
                p95_response_time = 0
                p99_response_time = 0
            
            # Build report
            report_data = {
                "user_statistics": user_stats,
                "performance_metrics": performance_stats,
                "engagement_metrics": engagement_stats,
                "summary": {
                    "report_month": report_month_str,
                    "generated_at": ist_now().isoformat(),
                    "total_users": user_stats.get("total_users", 0),
                    "active_users": user_stats.get("active_users_30d", 0),
                    "total_interactions": engagement_stats.get("total_interactions", 0),
                    "total_sessions": engagement_stats.get("total_sessions", 0),
                    "avg_response_time_ms": round(avg_response_time, 2),
                    "p95_response_time_ms": round(p95_response_time, 2),
                    "p99_response_time_ms": round(p99_response_time, 2),
                    "error_rate": performance_stats.get("error_rate", 0),
                    "fallback_rate": engagement_stats.get("fallback_rate", 0),
                    "feedback_stats": engagement_stats.get("feedback_stats", {})
                }
            }
            
            # Save report to database
            monthly_report = MonthlyReport(
                report_month=report_month_str,
                generated_at=ist_now(),
                total_users=user_stats.get("total_users", 0),
                active_users=user_stats.get("active_users_30d", 0),
                total_interactions=engagement_stats.get("total_interactions", 0),
                total_sessions=engagement_stats.get("total_sessions", 0),
                avg_response_time_ms=round(avg_response_time, 2),
                p95_response_time_ms=round(p95_response_time, 2),
                p99_response_time_ms=round(p99_response_time, 2),
                error_rate=performance_stats.get("error_rate", 0),
                fallback_rate=engagement_stats.get("fallback_rate", 0),
                feedback_stats=engagement_stats.get("feedback_stats", {}),
                top_queries=engagement_stats.get("top_queries", [])[:10],
                peak_usage_hours=engagement_stats.get("peak_usage_hours", []),
                report_data=report_data
            )
            
            await monthly_report.insert()
            logger.info(f"✅ Monthly report generated for {report_month_str}")
            
            return report_data
            
        except Exception as e:
            logger.error(f"Failed to generate monthly report: {e}", exc_info=True)
            raise
