"""
OpenTelemetry tracing configuration.

Provides distributed tracing for the perception API.
"""

from typing import Optional

from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_tracing(app: FastAPI, endpoint: str) -> None:
    """
    Configure OpenTelemetry tracing for the application.
    
    Args:
        app: FastAPI application instance
        endpoint: OTLP endpoint URL
    """
    # Create resource with service information
    resource = Resource.create({
        "service.name": "sam3-perception-api",
        "service.version": "0.1.0",
    })
    
    # Create tracer provider
    provider = TracerProvider(resource=resource)
    
    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=endpoint,
        insecure=True,  # Use TLS in production
    )
    
    # Add batch processor
    provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )
    
    # Set as global tracer provider
    trace.set_tracer_provider(provider)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)


def get_tracer(name: str = __name__) -> trace.Tracer:
    """Get a tracer instance for manual instrumentation."""
    return trace.get_tracer(name)


def create_span(name: str, attributes: Optional[dict] = None):
    """Create a new span with optional attributes."""
    tracer = get_tracer()
    span = tracer.start_span(name)
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)
    return span
